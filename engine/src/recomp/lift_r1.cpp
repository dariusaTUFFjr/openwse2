// R1 lifter core: context helpers, control flow, the translator module. See `lift_r1.h` and
// the conventions in `lift_r1_ctx.h`.

#include "lift_r1.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>

#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>

#include "lift_r1_ctx.h"
#include "llvm_common.h"

namespace wse::recomp {

std::string r1_symbol(std::uint64_t va) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "wse_fn_%llx", static_cast<unsigned long long>(va));
    return buf;
}

std::string r1_fast_symbol(std::uint64_t va) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "wse_fast_%llx", static_cast<unsigned long long>(va));
    return buf;
}

namespace r1 {
namespace {

constexpr std::size_t kFlagOffset[kFlagCount] = {cpu_offset::cf, cpu_offset::pf, cpu_offset::af,
                                                 cpu_offset::zf, cpu_offset::sf, cpu_offset::of,
                                                 cpu_offset::df};

bool is_high_byte(ZydisRegister r) {
    return r == ZYDIS_REGISTER_AH || r == ZYDIS_REGISTER_CH || r == ZYDIS_REGISTER_DH ||
           r == ZYDIS_REGISTER_BH;
}

unsigned reg_bits(ZydisRegister r) {
    return static_cast<unsigned>(ZydisRegisterGetWidth(ZYDIS_MACHINE_MODE_LONG_64, r));
}

}  // namespace

Ctx::Ctx(llvm::Module& m, llvm::Function* f, const DFunction& fn_in, const ImageInfo& image_in)
    : C(m.getContext()), M(m), B(m.getContext()), F(f), fn(fn_in), image(image_in) {
    state = F->getArg(0);  // slots are created by the caller once the entry block exists
}

int Ctx::gpr_index(ZydisRegister r) {
    if (ZydisRegisterGetClass(r) != ZYDIS_REGCLASS_GPR8 && ZydisRegisterGetClass(r) != ZYDIS_REGCLASS_GPR16 &&
        ZydisRegisterGetClass(r) != ZYDIS_REGCLASS_GPR32 && ZydisRegisterGetClass(r) != ZYDIS_REGCLASS_GPR64) {
        return -1;
    }
    const ZydisRegister w = ZydisRegisterGetLargestEnclosing(ZYDIS_MACHINE_MODE_LONG_64, r);
    if (w < ZYDIS_REGISTER_RAX || w > ZYDIS_REGISTER_R15) {
        return -1;
    }
    return static_cast<int>(w - ZYDIS_REGISTER_RAX);
}

int Ctx::xmm_index(ZydisRegister r) {
    if (r >= ZYDIS_REGISTER_XMM0 && r <= ZYDIS_REGISTER_XMM15) {
        return static_cast<int>(r - ZYDIS_REGISTER_XMM0);
    }
    return -1;
}

llvm::Value* Ctx::gpr64(int index) { return B.CreateLoad(B.getInt64Ty(), gpr_slot[index]); }
void Ctx::set_gpr64(int index, llvm::Value* v) { B.CreateStore(B.CreateZExtOrTrunc(v, B.getInt64Ty()), gpr_slot[index]); }

llvm::Value* Ctx::gpr(ZydisRegister r) {
    const int idx = gpr_index(r);
    llvm::Value* v = gpr64(idx);
    if (is_high_byte(r)) {
        return B.CreateTrunc(B.CreateLShr(v, 8), B.getInt8Ty());
    }
    return B.CreateTrunc(v, ity(reg_bits(r)));
}

void Ctx::set_gpr(ZydisRegister r, llvm::Value* v) {
    const int idx = gpr_index(r);
    const unsigned w = is_high_byte(r) ? 8 : reg_bits(r);
    v = B.CreateZExtOrTrunc(v, ity(w));
    if (w == 64 || w == 32) {
        set_gpr64(idx, B.CreateZExt(v, B.getInt64Ty()));  // 32-bit writes zero the upper half
        return;
    }
    llvm::Value* old = gpr64(idx);
    const unsigned shift = is_high_byte(r) ? 8 : 0;
    const std::uint64_t mask = ((w == 8 ? 0xFFULL : 0xFFFFULL) << shift);
    llvm::Value* merged = B.CreateOr(B.CreateAnd(old, B.getInt64(~mask)),
                                     B.CreateShl(B.CreateZExt(v, B.getInt64Ty()), shift));
    set_gpr64(idx, merged);
}

llvm::Value* Ctx::xmm_vec(int index) { return B.CreateLoad(v2i64(), xmm_slot[index]); }
llvm::Value* Ctx::xmm(int index) { return B.CreateBitCast(xmm_vec(index), i128()); }
void Ctx::set_xmm(int index, llvm::Value* v) {
    llvm::Type* ty = v->getType();
    if (ty->isVectorTy()) {
        B.CreateStore(B.CreateBitCast(v, v2i64()), xmm_slot[index]);
        return;
    }
    if (ty->isIntegerTy(32)) {
        auto* v4 = llvm::FixedVectorType::get(B.getInt32Ty(), 4);
        llvm::Value* ins = B.CreateInsertElement(llvm::ConstantAggregateZero::get(v4), v, std::uint64_t{0});
        B.CreateStore(B.CreateBitCast(ins, v2i64()), xmm_slot[index]);
        return;
    }
    if (ty->isIntegerTy(64)) {
        llvm::Value* ins = B.CreateInsertElement(llvm::ConstantAggregateZero::get(v2i64()), v, std::uint64_t{0});
        B.CreateStore(ins, xmm_slot[index]);
        return;
    }
    B.CreateStore(B.CreateBitCast(B.CreateZExtOrTrunc(v, i128()), v2i64()), xmm_slot[index]);
}

llvm::Value* Ctx::flag(Flag f) { return B.CreateLoad(B.getInt1Ty(), flag_slot[f]); }
void Ctx::set_flag(Flag f, llvm::Value* v) { B.CreateStore(B.CreateTrunc(v, B.getInt1Ty()), flag_slot[f]); }

void Ctx::set_szp(llvm::Value* r) {
    llvm::Type* t = r->getType();
    set_flag(ZF, B.CreateICmpEQ(r, llvm::ConstantInt::get(t, 0)));
    set_flag(SF, B.CreateICmpSLT(r, llvm::ConstantInt::get(t, 0)));
    llvm::Value* low = B.CreateTrunc(r, B.getInt8Ty());
    llvm::Value* pop = B.CreateUnaryIntrinsic(llvm::Intrinsic::ctpop, low);
    set_flag(PF, B.CreateICmpEQ(B.CreateAnd(pop, B.getInt8(1)), B.getInt8(0)));
}

void Ctx::set_add_flags(llvm::Value* a, llvm::Value* b, llvm::Value* r, llvm::Value* cin) {
    const unsigned n = a->getType()->getIntegerBitWidth();
    llvm::Type* wide = ity(n + 1);
    llvm::Value* s = B.CreateAdd(B.CreateZExt(a, wide), B.CreateZExt(b, wide));
    if (cin != nullptr) {
        s = B.CreateAdd(s, B.CreateZExt(cin, wide));
    }
    set_flag(CF, B.CreateTrunc(B.CreateLShr(s, n), B.getInt1Ty()));
    llvm::Value* zero = llvm::ConstantInt::get(a->getType(), 0);
    set_flag(OF, B.CreateICmpSLT(B.CreateAnd(B.CreateXor(a, r), B.CreateXor(b, r)), zero));
    set_flag(AF, B.CreateICmpNE(B.CreateAnd(B.CreateXor(B.CreateXor(a, b), r),
                                            llvm::ConstantInt::get(a->getType(), 0x10)),
                                zero));
    set_szp(r);
}

void Ctx::set_sub_flags(llvm::Value* a, llvm::Value* b, llvm::Value* r, llvm::Value* bin) {
    const unsigned n = a->getType()->getIntegerBitWidth();
    llvm::Type* wide = ity(n + 1);
    llvm::Value* d = B.CreateSub(B.CreateZExt(a, wide), B.CreateZExt(b, wide));
    if (bin != nullptr) {
        d = B.CreateSub(d, B.CreateZExt(bin, wide));
    }
    set_flag(CF, B.CreateTrunc(B.CreateLShr(d, n), B.getInt1Ty()));
    llvm::Value* zero = llvm::ConstantInt::get(a->getType(), 0);
    set_flag(OF, B.CreateICmpSLT(B.CreateAnd(B.CreateXor(a, b), B.CreateXor(a, r)), zero));
    set_flag(AF, B.CreateICmpNE(B.CreateAnd(B.CreateXor(B.CreateXor(a, b), r),
                                            llvm::ConstantInt::get(a->getType(), 0x10)),
                                zero));
    set_szp(r);
}

void Ctx::set_logic_flags(llvm::Value* r) {
    set_flag(CF, B.getFalse());
    set_flag(OF, B.getFalse());
    set_szp(r);
}

llvm::Value* Ctx::condition(ZydisMnemonic m) {
    std::string s = ZydisMnemonicGetString(m);
    for (const char* p : {"cmov", "set", "j"}) {
        if (s.rfind(p, 0) == 0) {
            s = s.substr(std::strlen(p));
            break;
        }
    }
    auto ne = [&](llvm::Value* v) { return B.CreateNot(v); };
    if (s == "o") return flag(OF);
    if (s == "no") return ne(flag(OF));
    if (s == "b") return flag(CF);
    if (s == "nb") return ne(flag(CF));
    if (s == "z") return flag(ZF);
    if (s == "nz") return ne(flag(ZF));
    if (s == "be") return B.CreateOr(flag(CF), flag(ZF));
    if (s == "nbe") return ne(B.CreateOr(flag(CF), flag(ZF)));
    if (s == "s") return flag(SF);
    if (s == "ns") return ne(flag(SF));
    if (s == "p") return flag(PF);
    if (s == "np") return ne(flag(PF));
    if (s == "l") return B.CreateICmpNE(flag(SF), flag(OF));
    if (s == "nl") return B.CreateICmpEQ(flag(SF), flag(OF));
    if (s == "le") return B.CreateOr(flag(ZF), B.CreateICmpNE(flag(SF), flag(OF)));
    if (s == "nle") return B.CreateAnd(ne(flag(ZF)), B.CreateICmpEQ(flag(SF), flag(OF)));
    return nullptr;
}

llvm::Value* Ctx::state_ptr(std::size_t offset) {
    return B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), state, offset);
}

llvm::Value* Ctx::address(const DInsn& d, const ZydisDecodedOperand& o) {
    const bool no_seg = o.mem.segment != ZYDIS_REGISTER_FS && o.mem.segment != ZYDIS_REGISTER_GS;
    const int base_idx = gpr_index(o.mem.base);
    // `rsp`-based operands are the thread's own stack in every mode (as before R9); registers
    // derived from it (`lea`/`mov` copies, a frame-pointer `rbp`) count too when relaxation is on.
    last_address_stack =
        no_seg && base_idx >= 0 &&
        (base_idx == 4 || (image.relaxed_tso && (stack_reg_mask & (1U << base_idx)) != 0));
    last_address_readonly = false;
    if (no_seg && o.mem.base == ZYDIS_REGISTER_RIP && o.mem.index == ZYDIS_REGISTER_NONE) {
        const std::uint64_t target =
            d.next() + (o.mem.disp.has_displacement ? static_cast<std::uint64_t>(o.mem.disp.value) : 0ULL);
        for (const AddrRange& r : image.readonly_ranges) {
            if (target >= r.begin && target < r.end) {
                last_address_readonly = true;
                break;
            }
        }
    }
    llvm::Value* a = B.getInt64(o.mem.disp.has_displacement ? static_cast<std::uint64_t>(o.mem.disp.value) : 0);
    if (o.mem.base == ZYDIS_REGISTER_RIP) {
        a = B.CreateAdd(a, B.getInt64(d.next()));
    } else if (o.mem.base != ZYDIS_REGISTER_NONE) {
        a = B.CreateAdd(a, B.CreateZExt(gpr(o.mem.base), B.getInt64Ty()));
    }
    if (o.mem.index != ZYDIS_REGISTER_NONE) {
        a = B.CreateAdd(a, B.CreateMul(B.CreateZExt(gpr(o.mem.index), B.getInt64Ty()), B.getInt64(o.mem.scale)));
    }
    if (d.ins.address_width == 32) {
        a = B.CreateAnd(a, B.getInt64(0xFFFFFFFFULL));
    }
    if (o.mem.segment == ZYDIS_REGISTER_FS || o.mem.segment == ZYDIS_REGISTER_GS) {
        flush();
        llvm::Value* base = call_hook(offsetof(RuntimeHooks, segment_base),
                                      {B.getInt32(o.mem.segment == ZYDIS_REGISTER_FS ? 4 : 5)},
                                      {B.getInt32Ty()}, B.getInt64Ty());
        reload();
        a = B.CreateAdd(a, base);
    }
    return a;
}

llvm::Value* Ctx::host_ptr(llvm::Value* addr64) { return B.CreateIntToPtr(addr64, ptr()); }

namespace {

bool is_stack(Mem kind, bool last_address_stack) {
    return kind == Mem::stack || (kind == Mem::automatic && last_address_stack);
}

}  // namespace

// Shared accesses keep x86's order on AArch64 (`docs/RECOMP_PLAN.md` "R3 design"): a naturally
// aligned 2/4/8-byte load is an acquire load (LDAPR/LDAR) and a store a release store (STLR); bytes
// are always aligned; a misaligned access, or a wider one (SSE), is plain with a barrier after a
// load (`dmb ishld`) or before a store (`dmb ish`). On x86 all of this compiles to plain moves.
llvm::Value* Ctx::load(llvm::Value* addr64, unsigned bits_in, Mem kind) {
    const bool stack = is_stack(kind, last_address_stack);
    const bool is_ro = (kind == Mem::automatic && last_address_readonly);
    last_address_stack = false;
    last_address_readonly = false;
    llvm::Type* t = ity(bits_in);
    if (kind != Mem::shared && (stack || is_ro || fn_relaxed_tso)) {
        llvm::LoadInst* l = B.CreateLoad(t, host_ptr(addr64));
        l->setAlignment(llvm::Align(1));
        // A relaxed shared load is still a load the x86 code performs every time: volatile keeps
        // LLVM from hoisting it out of a loop (a spin on a flag another thread sets would never
        // end) while emitting the same plain LDR with no barrier.
        if (!stack && !is_ro) l->setVolatile(true);
        return l;
    }
    if (bits_in == 8) {
        llvm::LoadInst* l = B.CreateLoad(t, host_ptr(addr64));
        l->setAlignment(llvm::Align(1));
        l->setAtomic(llvm::AtomicOrdering::Acquire);
        return l;
    }
    if (bits_in != 16 && bits_in != 32 && bits_in != 64) {
        llvm::LoadInst* l = B.CreateLoad(t, host_ptr(addr64));
        l->setAlignment(llvm::Align(1));
        B.CreateFence(llvm::AtomicOrdering::Acquire);
        return l;
    }
    llvm::Value* aligned = B.CreateICmpEQ(B.CreateAnd(addr64, B.getInt64(bits_in / 8 - 1)), B.getInt64(0));
    llvm::BasicBlock* from = B.GetInsertBlock();
    llvm::BasicBlock* atomic_bb = llvm::BasicBlock::Create(C, "ld_acq", F);
    llvm::BasicBlock* plain_bb = llvm::BasicBlock::Create(C, "ld_unaligned", F);
    llvm::BasicBlock* done = llvm::BasicBlock::Create(C, "ld_done", F);
    B.CreateCondBr(aligned, atomic_bb, plain_bb);
    (void)from;
    B.SetInsertPoint(atomic_bb);
    llvm::LoadInst* a = B.CreateLoad(t, host_ptr(addr64));
    a->setAlignment(llvm::Align(bits_in / 8));
    a->setAtomic(llvm::AtomicOrdering::Acquire);
    B.CreateBr(done);
    B.SetInsertPoint(plain_bb);
    llvm::LoadInst* p = B.CreateLoad(t, host_ptr(addr64));
    p->setAlignment(llvm::Align(1));
    B.CreateFence(llvm::AtomicOrdering::Acquire);
    B.CreateBr(done);
    B.SetInsertPoint(done);
    llvm::PHINode* phi = B.CreatePHI(t, 2);
    phi->addIncoming(a, atomic_bb);
    phi->addIncoming(p, plain_bb);
    return phi;
}

llvm::Value* Ctx::load_vec128(llvm::Value* addr64, Mem kind) {
    const bool stack = is_stack(kind, last_address_stack);
    const bool is_ro = (kind == Mem::automatic && last_address_readonly);
    last_address_stack = false;
    last_address_readonly = false;
    llvm::LoadInst* l = B.CreateLoad(v2i64(), host_ptr(addr64));
    l->setAlignment(llvm::Align(1));
    if (kind == Mem::shared || (!stack && !is_ro && !fn_relaxed_tso)) {
        B.CreateFence(llvm::AtomicOrdering::Acquire);
    } else if (!stack && !is_ro) {
        l->setVolatile(true);  // relaxed shared load: see `load()`
    }
    return l;
}

void Ctx::store(llvm::Value* addr64, llvm::Value* v, Mem kind) {
    const bool stack = is_stack(kind, last_address_stack);
    last_address_stack = false;
    last_address_readonly = false;
    const unsigned bits_in = v->getType()->getPrimitiveSizeInBits();
    if (kind != Mem::shared && (stack || fn_relaxed_tso)) {
        llvm::StoreInst* s = B.CreateStore(v, host_ptr(addr64));
        s->setAlignment(llvm::Align(1));
        // A relaxed shared store happens where the x86 code does it: volatile keeps LLVM from
        // sinking it out of a loop (a progress counter another thread polls), no barrier added.
        if (!stack) s->setVolatile(true);
        return;
    }
    if (!v->getType()->isIntegerTy()) {
        B.CreateFence(llvm::AtomicOrdering::Release);
        llvm::StoreInst* s = B.CreateStore(v, host_ptr(addr64));
        s->setAlignment(llvm::Align(1));
        return;
    }
    if (bits_in == 8) {
        llvm::StoreInst* s = B.CreateStore(v, host_ptr(addr64));
        s->setAlignment(llvm::Align(1));
        s->setAtomic(llvm::AtomicOrdering::Release);
        return;
    }
    if (bits_in != 16 && bits_in != 32 && bits_in != 64) {
        B.CreateFence(llvm::AtomicOrdering::Release);
        llvm::StoreInst* s = B.CreateStore(v, host_ptr(addr64));
        s->setAlignment(llvm::Align(1));
        return;
    }
    llvm::Value* aligned = B.CreateICmpEQ(B.CreateAnd(addr64, B.getInt64(bits_in / 8 - 1)), B.getInt64(0));
    llvm::BasicBlock* atomic_bb = llvm::BasicBlock::Create(C, "st_rel", F);
    llvm::BasicBlock* plain_bb = llvm::BasicBlock::Create(C, "st_unaligned", F);
    llvm::BasicBlock* done = llvm::BasicBlock::Create(C, "st_done", F);
    B.CreateCondBr(aligned, atomic_bb, plain_bb);
    B.SetInsertPoint(atomic_bb);
    llvm::StoreInst* a = B.CreateStore(v, host_ptr(addr64));
    a->setAlignment(llvm::Align(bits_in / 8));
    a->setAtomic(llvm::AtomicOrdering::Release);
    B.CreateBr(done);
    B.SetInsertPoint(plain_bb);
    B.CreateFence(llvm::AtomicOrdering::Release);
    llvm::StoreInst* p = B.CreateStore(v, host_ptr(addr64));
    p->setAlignment(llvm::Align(1));
    B.CreateBr(done);
    B.SetInsertPoint(done);
}

llvm::Value* Ctx::read(const DInsn& d, int i) {
    const ZydisDecodedOperand& o = d.ops[i];
    switch (o.type) {
    case ZYDIS_OPERAND_TYPE_REGISTER: {
        const int x = xmm_index(o.reg.value);
        if (x >= 0) {
            if (o.size == 64) {
                return B.CreateExtractElement(xmm_vec(x), std::uint64_t{0});
            }
            if (o.size == 32) {
                auto* v4 = llvm::FixedVectorType::get(B.getInt32Ty(), 4);
                return B.CreateExtractElement(B.CreateBitCast(xmm_vec(x), v4), std::uint64_t{0});
            }
            return B.CreateTrunc(xmm(x), ity(o.size));
        }
        return gpr(o.reg.value);
    }
    case ZYDIS_OPERAND_TYPE_MEMORY:
        return load(address(d, o), o.size);
    case ZYDIS_OPERAND_TYPE_IMMEDIATE: {
        unsigned n = o.size == 0 ? 64 : o.size;
        // A signed immediate narrower than the destination (`mov r64, imm32`, `add r64, imm8`)
        // is sign-extended to it; Zydis already sign-extends `imm.value` to 64 bits.
        if (i > 0 && o.imm.is_signed && d.ops[0].type != ZYDIS_OPERAND_TYPE_IMMEDIATE &&
            d.ops[0].size > n) {
            n = d.ops[0].size;
        }
        const std::uint64_t mask = n >= 64 ? ~0ULL : ((1ULL << n) - 1ULL);
        return llvm::ConstantInt::get(ity(n), o.imm.value.u & mask);
    }
    default:
        break;
    }
    return llvm::ConstantInt::get(ity(o.size == 0 ? 64 : o.size), 0);
}

void Ctx::write(const DInsn& d, int i, llvm::Value* v) {
    const ZydisDecodedOperand& o = d.ops[i];
    if (o.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        const int x = xmm_index(o.reg.value);
        if (x >= 0) {
            if (o.size >= 128) {
                set_xmm(x, v);
            } else if (o.size == 64) {
                set_xmm(x, B.CreateInsertElement(xmm_vec(x), B.CreateZExtOrTrunc(v, B.getInt64Ty()), std::uint64_t{0}));
            } else if (o.size == 32) {
                auto* v4 = llvm::FixedVectorType::get(B.getInt32Ty(), 4);
                set_xmm(x, B.CreateInsertElement(B.CreateBitCast(xmm_vec(x), v4),
                                                 B.CreateZExtOrTrunc(v, B.getInt32Ty()), std::uint64_t{0}));
            } else {
                // Narrow XMM writes merge into the low bits (families decide when to zero).
                const std::uint64_t n = o.size;
                llvm::Value* low = B.CreateZExt(B.CreateZExtOrTrunc(v, ity(o.size)), i128());
                llvm::Value* keep = B.CreateShl(B.CreateLShr(xmm(x), n), n);
                set_xmm(x, B.CreateOr(keep, low));
            }
            return;
        }
        set_gpr(o.reg.value, v);
        return;
    }
    if (o.type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (v->getType()->isVectorTy()) {
            v = B.CreateBitCast(v, ity(o.size));
        }
        store(address(d, o), B.CreateZExtOrTrunc(v, ity(o.size)));
    }
}

// A register-passing (`wse_fast_<va>`) body owns only the registers that ABI passes: rax, rcx,
// rdx, rsp, r8-r11 and xmm0-xmm3 (`is_fast_abi_eligible` rejects any other). Its other slots are
// placeholders, never the caller's values, so a flush inside it (`div128`, a fault) must not
// write them: the caller's rbx/rsi/rdi/rbp/r12-r15/xmm4-15 may live in the CpuState and would be
// clobbered for a caller that reloads the whole state (through the `wse_fn_<va>` thunk).
bool Ctx::owns_gpr(int r) const { return !is_fast_abi || ((kFastGprMask >> r) & 1U) != 0U; }
bool Ctx::owns_xmm(int r) const { return !is_fast_abi || r < 4; }

void Ctx::flush() {
    for (int r = 0; r < 16; ++r) {
        if (owns_gpr(r)) {
            B.CreateStore(B.CreateLoad(B.getInt64Ty(), gpr_slot[r]), state_ptr(cpu_offset::gpr + 8U * r));
        }
        if (owns_xmm(r)) {
            llvm::StoreInst* s =
                B.CreateStore(B.CreateLoad(v2i64(), xmm_slot[r]), state_ptr(cpu_offset::xmm + 16U * r));
            s->setAlignment(llvm::Align(16));
        }
    }
    for (int k = 0; k < kFlagCount; ++k) {
        B.CreateStore(B.CreateZExt(B.CreateLoad(B.getInt1Ty(), flag_slot[k]), B.getInt8Ty()),
                      state_ptr(kFlagOffset[k]));
    }
}

void Ctx::reload() {
    for (int r = 0; r < 16; ++r) {
        if (owns_gpr(r)) {
            B.CreateStore(B.CreateLoad(B.getInt64Ty(), state_ptr(cpu_offset::gpr + 8U * r)), gpr_slot[r]);
        }
        if (owns_xmm(r)) {
            llvm::LoadInst* x = B.CreateLoad(v2i64(), state_ptr(cpu_offset::xmm + 16U * r));
            x->setAlignment(llvm::Align(16));
            B.CreateStore(x, xmm_slot[r]);
        }
    }
    for (int k = 0; k < kFlagCount; ++k) {
        llvm::Value* byte = B.CreateLoad(B.getInt8Ty(), state_ptr(kFlagOffset[k]));
        B.CreateStore(B.CreateICmpNE(byte, B.getInt8(0)), flag_slot[k]);
    }
}

llvm::CallInst* Ctx::call_hook(std::size_t offset, llvm::ArrayRef<llvm::Value*> extra,
                               llvm::ArrayRef<llvm::Type*> extra_types, llvm::Type* ret) {
    llvm::Value* hooks = B.CreateLoad(ptr(), state_ptr(cpu_offset::hooks));
    llvm::Value* slot = B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), hooks, offset);
    llvm::Value* fp = B.CreateLoad(ptr(), slot);
    std::vector<llvm::Type*> params{ptr()};
    params.insert(params.end(), extra_types.begin(), extra_types.end());
    auto* fty = llvm::FunctionType::get(ret != nullptr ? ret : B.getVoidTy(), params, false);
    std::vector<llvm::Value*> args{state};
    args.insert(args.end(), extra.begin(), extra.end());
    return B.CreateCall(fty, fp, args);
}

void Ctx::emit_unsupported(std::uint64_t va, const std::string& what) {
    flush();
    B.CreateStore(B.getInt64(va), state_ptr(cpu_offset::rip));
    llvm::Constant* text = B.CreateGlobalStringPtr(what, "why");
    call_hook(offsetof(RuntimeHooks, unsupported), {B.getInt64(va), text}, {B.getInt64Ty(), ptr()});
    if (is_fast_abi) {
        // Hand back the registers as they are (a zeroed aggregate would set the caller's rsp to 0)
        // and the rip the fault left (`kFaultRip`), so the caller's return check stays quiet.
        static constexpr int kRet[8] = {0, 1, 2, 8, 9, 10, 11, 4};
        llvm::Value* ret = llvm::UndefValue::get(F->getReturnType());
        for (unsigned i = 0; i < 8U; ++i) ret = B.CreateInsertValue(ret, gpr64(kRet[i]), {i});
        for (unsigned i = 0; i < 4U; ++i) ret = B.CreateInsertValue(ret, xmm_vec(static_cast<int>(i)), {8U + i});
        ret = B.CreateInsertValue(ret, B.CreateLoad(B.getInt64Ty(), state_ptr(cpu_offset::rip)), {12U});
        B.CreateRet(ret);
    } else {
        B.CreateRetVoid();  // the hook does not return in practice; returning keeps the IR valid
    }
}

}  // namespace r1

// ---------------------------------------------------------------------------------------------

namespace {

constexpr int kFastArgGprs[8] = {0, 1, 2, 8, 9, 10, 11, 4};  // rax, rcx, rdx, r8, r9, r10, r11, rsp
constexpr int kFastRetGprs[8] = {0, 1, 2, 8, 9, 10, 11, 4};  // rax, rcx, rdx, r8, r9, r10, r11, rsp

}  // namespace

struct R1Translator::Impl {
    ImageInfo image;
    std::unique_ptr<llvm::LLVMContext> context = std::make_unique<llvm::LLVMContext>();
    std::unique_ptr<llvm::Module> module;
    std::map<std::uint64_t, llvm::Function*> functions;      // declared (some defined)
    std::map<std::uint64_t, llvm::Function*> fast_functions; // defined wse_fast_<va>
    std::map<std::uint64_t, llvm::Function*> inline_leaves;  // cloned external inline leaves
    std::set<std::uint64_t> light;                           // translated without optimisation
    // Direct cross-batch calls (`wse_ext_calls[]`): writable pointer slots patched at startup
    // by `Runtime::patch_external_calls` so hot cross-batch edges skip `Runtime::hook_dispatch`.
    std::map<std::uint64_t, std::size_t> ext_call_index;
    std::vector<std::uint64_t> ext_call_vas;
    llvm::GlobalVariable* ext_calls_gv = nullptr;
    // Fast-ABI cross-batch calls (`wse_ext_fast_calls[]`).
    std::map<std::uint64_t, std::size_t> ext_fast_index;
    std::vector<std::uint64_t> ext_fast_vas;
    llvm::GlobalVariable* ext_fast_gv = nullptr;
    std::string init_error;
    LiftReport report;
    double lift_ms = 0.0;
    double total_ms = 0.0;  // lift time of discarded / emitted modules

    void reset() {
        module = std::make_unique<llvm::Module>("wse_r1", *context);
        functions.clear();
        fast_functions.clear();
        inline_leaves.clear();
        ext_call_index.clear();
        ext_call_vas.clear();
        ext_calls_gv = nullptr;
        ext_fast_index.clear();
        ext_fast_vas.clear();
        ext_fast_gv = nullptr;
        lift_ms = 0.0;
    }

    llvm::StructType* fast_ret_ty() {
        llvm::Type* i64 = llvm::Type::getInt64Ty(*context);
        llvm::Type* v2 = llvm::FixedVectorType::get(i64, 2);
        return llvm::StructType::get(*context, {i64, i64, i64, i64, i64, i64, i64, i64, v2, v2, v2, v2, i64});
    }

    llvm::FunctionType* fast_fn_ty() {
        llvm::Type* p = llvm::PointerType::getUnqual(*context);
        llvm::Type* i64 = llvm::Type::getInt64Ty(*context);
        llvm::Type* v2 = llvm::FixedVectorType::get(i64, 2);
        return llvm::FunctionType::get(fast_ret_ty(), {p, i64, i64, i64, i64, i64, i64, i64, i64, v2, v2, v2, v2},
                                       false);
    }

    llvm::Value* ext_call_slot_ptr(llvm::IRBuilder<>& b, std::uint64_t va) {
        auto [it, fresh] = ext_call_index.emplace(va, ext_call_vas.size());
        if (fresh) {
            ext_call_vas.push_back(va);
        }
        if (ext_calls_gv == nullptr) {
            auto* arr_ty = llvm::ArrayType::get(llvm::PointerType::getUnqual(*context), 0);
            ext_calls_gv = new llvm::GlobalVariable(*module, arr_ty, false, llvm::GlobalValue::ExternalLinkage,
                                                    nullptr, "wse_ext_calls_placeholder");
            ext_calls_gv->setVisibility(llvm::GlobalValue::ProtectedVisibility);
            ext_calls_gv->setDSOLocal(true);
            ext_calls_gv->setSection(".data");
        }
        return b.CreateConstInBoundsGEP1_64(llvm::PointerType::getUnqual(*context), ext_calls_gv, it->second);
    }

    llvm::Value* ext_fast_slot_ptr(llvm::IRBuilder<>& b, std::uint64_t va) {
        auto [it, fresh] = ext_fast_index.emplace(va, ext_fast_vas.size());
        if (fresh) {
            ext_fast_vas.push_back(va);
        }
        if (ext_fast_gv == nullptr) {
            auto* arr_ty = llvm::ArrayType::get(llvm::PointerType::getUnqual(*context), 0);
            ext_fast_gv = new llvm::GlobalVariable(*module, arr_ty, false, llvm::GlobalValue::ExternalLinkage,
                                                   nullptr, "wse_ext_fast_placeholder");
            ext_fast_gv->setVisibility(llvm::GlobalValue::ProtectedVisibility);
            ext_fast_gv->setDSOLocal(true);
            ext_fast_gv->setSection(".data");
        }
        return b.CreateConstInBoundsGEP1_64(llvm::PointerType::getUnqual(*context), ext_fast_gv, it->second);
    }

    llvm::FunctionType* guest_fn_ty() {
        return llvm::FunctionType::get(llvm::Type::getVoidTy(*context),
                                       {llvm::PointerType::getUnqual(*context)}, false);
    }

    llvm::Function* function(std::uint64_t va) {
        auto it = functions.find(va);
        if (it != functions.end()) {
            return it->second;
        }
        auto* fty = llvm::FunctionType::get(llvm::Type::getVoidTy(*context),
                                            {llvm::PointerType::getUnqual(*context)}, false);
        auto* f = llvm::Function::Create(fty, llvm::Function::ExternalLinkage, r1_symbol(va), *module);
        // Exported, but bound inside the module: calls between translated functions are direct (no
        // PLT) and `wse_fn_table` needs only relative relocations (the own ELF loader's subset).
        f->setVisibility(llvm::GlobalValue::ProtectedVisibility);
        f->setDSOLocal(true);
        f->addFnAttr(llvm::Attribute::NoUnwind);
        functions.emplace(va, f);
        return f;
    }

    static bool lift_body(Impl& impl, const DFunction& fn, llvm::Function* f, bool fast_abi, std::string* why);
};

R1Translator::R1Translator(const ImageInfo& image) : impl_(std::make_unique<Impl>()) {
    impl_->image = image;
    impl_->reset();
    const llvm_common::TargetSetup t = llvm_common::make_target(2);
    impl_->init_error = t.error;
}

R1Translator::~R1Translator() = default;

const std::string& R1Translator::init_error() const noexcept { return impl_->init_error; }
const LiftReport& R1Translator::report() const noexcept { return impl_->report; }
void R1Translator::discard() {
    impl_->total_ms += impl_->lift_ms;
    impl_->reset();
}
double R1Translator::total_lift_ms() const noexcept { return impl_->total_ms + impl_->lift_ms; }

void R1Translator::set_light(const std::set<std::uint64_t>& vas) {
    impl_->light = vas;
}

void R1Translator::declare(const std::vector<std::uint64_t>& entries) {
    for (const std::uint64_t va : entries) {
        (void)impl_->function(va);
    }
}

namespace {

using r1::Ctx;
using r1::Lifted;

// Pushes `ret_va`, runs `emit_call` (which must leave the state flushed and call the target),
// reloads, and reports a callee that did not return to `ret_va`.
// A guest call: pushes `ret_va`, calls, and checks the callee came back to `ret_va`. When it did
// not, either the runtime is unwinding (`CpuState::unwind_rsp`): this frame resumes through
// `resume` when the target `rsp` is its call-site `rsp` (a function without a resume switch
// faults there), and otherwise returns at once so the unwind continues outward; or it is a real
// bad return, reported to the hook. The code that does not throw pays only the existing compare.
template <typename Emit>
void guest_call(Ctx& c, std::uint64_t ret_va, llvm::BasicBlock* resume, Emit&& emit_call) {
    llvm::Value* site_rsp = c.gpr64(4);
    llvm::Value* rsp = c.B.CreateSub(site_rsp, c.B.getInt64(8));
    c.set_gpr64(4, rsp);
    c.store(rsp, c.B.getInt64(ret_va), r1::Mem::stack);
    c.flush();
    emit_call();
    c.reload();
    llvm::Value* rip = c.B.CreateLoad(c.B.getInt64Ty(), c.state_ptr(cpu_offset::rip));
    llvm::BasicBlock* off_path = llvm::BasicBlock::Create(c.C, "ret_elsewhere", c.F);
    llvm::BasicBlock* unwinding = llvm::BasicBlock::Create(c.C, "unwinding", c.F);
    llvm::BasicBlock* here = llvm::BasicBlock::Create(c.C, "unwind_here", c.F);
    llvm::BasicBlock* outward = llvm::BasicBlock::Create(c.C, "unwind_outward", c.F);
    llvm::BasicBlock* bad = llvm::BasicBlock::Create(c.C, "bad_ret", c.F);
    llvm::BasicBlock* ok = llvm::BasicBlock::Create(c.C, "ret_ok", c.F);
    c.B.CreateCondBr(c.B.CreateICmpEQ(rip, c.B.getInt64(ret_va)), ok, off_path);
    c.B.SetInsertPoint(off_path);
    llvm::Value* target = c.B.CreateLoad(c.B.getInt64Ty(), c.state_ptr(cpu_offset::unwind_rsp));
    c.B.CreateCondBr(c.B.CreateICmpNE(target, c.B.getInt64(0)), unwinding, bad);
    c.B.SetInsertPoint(unwinding);
    c.B.CreateCondBr(c.B.CreateICmpEQ(target, site_rsp), here, outward);
    c.B.SetInsertPoint(outward);
    c.B.CreateRetVoid();  // the state already holds the target's context: just return
    c.B.SetInsertPoint(here);
    c.B.CreateStore(c.B.getInt64(0), c.state_ptr(cpu_offset::unwind_rsp));
    if (resume != nullptr) {
        c.B.CreateBr(resume);
    } else {
        c.emit_unsupported(ret_va, "unwind into a frame with no exception handler");
    }
    c.B.SetInsertPoint(bad);
    c.call_hook(offsetof(RuntimeHooks, bad_return), {c.B.getInt64(ret_va)}, {c.B.getInt64Ty()});
    c.B.CreateBr(ok);
    c.B.SetInsertPoint(ok);
}

llvm::CallInst* call_dispatch(Ctx& c, llvm::Value* target) {
    return c.call_hook(offsetof(RuntimeHooks, dispatch), {target}, {c.B.getInt64Ty()});
}

bool function_has_sync(const DFunction& fn, const ImageInfo& image) {
    for (const auto& [va, d] : fn.insns) {
        (void)va;
        if ((d.ins.attributes & ZYDIS_ATTRIB_HAS_LOCK) != 0) return true;
        const auto mn = d.ins.mnemonic;
        if (mn == ZYDIS_MNEMONIC_MFENCE || mn == ZYDIS_MNEMONIC_SFENCE || mn == ZYDIS_MNEMONIC_LFENCE ||
            mn == ZYDIS_MNEMONIC_PAUSE || mn == ZYDIS_MNEMONIC_CMPXCHG || mn == ZYDIS_MNEMONIC_CMPXCHG8B ||
            mn == ZYDIS_MNEMONIC_CMPXCHG16B || mn == ZYDIS_MNEMONIC_XADD) {
            return true;
        }
        if (mn == ZYDIS_MNEMONIC_XCHG) {
            for (std::uint8_t i = 0; i < d.ins.operand_count_visible; ++i) {
                if (d.ops[i].type == ZYDIS_OPERAND_TYPE_MEMORY) return true;
            }
        }
        if (d.ins.meta.category == ZYDIS_CATEGORY_CALL || d.ins.meta.category == ZYDIS_CATEGORY_UNCOND_BR) {
            const ZydisDecodedOperand& o = d.ops[0];
            if (o.type == ZYDIS_OPERAND_TYPE_MEMORY && o.mem.base == ZYDIS_REGISTER_RIP &&
                o.mem.index == ZYDIS_REGISTER_NONE) {
                const std::uint64_t slot = d.next() + static_cast<std::uint64_t>(o.mem.disp.value);
                if (image.sync_iat_slots.count(slot) != 0) return true;
            }
        }
    }
    return false;
}

bool function_uses_rbp_frame(const DFunction& fn) {
    bool initialized_in_entry = false;
    std::uint64_t va = fn.entry;
    while (const DInsn* dp = fn.at(va)) {
        const DInsn& d = *dp;
        const auto mn = d.ins.mnemonic;
        if ((mn == ZYDIS_MNEMONIC_MOV && d.ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
             d.ops[0].reg.value == ZYDIS_REGISTER_RBP && d.ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER &&
             d.ops[1].reg.value == ZYDIS_REGISTER_RSP) ||
            (mn == ZYDIS_MNEMONIC_LEA && d.ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
             d.ops[0].reg.value == ZYDIS_REGISTER_RBP && d.ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY &&
             d.ops[1].mem.base == ZYDIS_REGISTER_RSP && d.ops[1].mem.index == ZYDIS_REGISTER_NONE &&
             d.ops[1].mem.segment != ZYDIS_REGISTER_FS && d.ops[1].mem.segment != ZYDIS_REGISTER_GS)) {
            initialized_in_entry = true;
            break;
        }
        // Allow initial prologue save `push rbp` before `mov/lea rbp, rsp`.
        if (mn == ZYDIS_MNEMONIC_PUSH && d.ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
            d.ops[0].reg.value == ZYDIS_REGISTER_RBP) {
            va = d.next();
            if (fn.block_starts.count(va) != 0) break;
            continue;
        }
        // Any other read/write of RBP before initialization disqualifies it.
        for (std::uint8_t i = 0; i < d.ins.operand_count; ++i) {
            const ZydisDecodedOperand& o = d.ops[i];
            if (o.type == ZYDIS_OPERAND_TYPE_REGISTER && Ctx::gpr_index(o.reg.value) == 5) return false;
            if (o.type == ZYDIS_OPERAND_TYPE_MEMORY &&
                (Ctx::gpr_index(o.mem.base) == 5 || Ctx::gpr_index(o.mem.index) == 5)) {
                return false;
            }
        }
        if (d.ins.meta.category == ZYDIS_CATEGORY_COND_BR || d.ins.meta.category == ZYDIS_CATEGORY_UNCOND_BR ||
            d.ins.meta.category == ZYDIS_CATEGORY_RET || d.ins.meta.category == ZYDIS_CATEGORY_CALL) {
            break;
        }
        va = d.next();
        if (fn.block_starts.count(va) != 0) break;
    }
    if (!initialized_in_entry) return false;
    for (const auto& [iva, d] : fn.insns) {
        (void)iva;
        const auto mn = d.ins.mnemonic;
        bool writes_rbp = false;
        for (std::uint8_t i = 0; i < d.ins.operand_count; ++i) {
            const ZydisDecodedOperand& o = d.ops[i];
            if ((o.actions & ZYDIS_OPERAND_ACTION_MASK_WRITE) != 0 && o.type == ZYDIS_OPERAND_TYPE_REGISTER &&
                Ctx::gpr_index(o.reg.value) == 5) {
                writes_rbp = true;
                break;
            }
        }
        if (!writes_rbp) continue;
        if (mn == ZYDIS_MNEMONIC_MOV && d.ops[0].reg.value == ZYDIS_REGISTER_RBP &&
            d.ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER && d.ops[1].reg.value == ZYDIS_REGISTER_RSP) {
            continue;
        }
        if (mn == ZYDIS_MNEMONIC_LEA && d.ops[0].reg.value == ZYDIS_REGISTER_RBP &&
            d.ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY && d.ops[1].mem.base == ZYDIS_REGISTER_RSP &&
            d.ops[1].mem.index == ZYDIS_REGISTER_NONE &&
            d.ops[1].mem.segment != ZYDIS_REGISTER_FS && d.ops[1].mem.segment != ZYDIS_REGISTER_GS) {
            continue;
        }
        // Allow epilogue pop rbp / leave only if followed by ret without further RBP memory use.
        if (mn == ZYDIS_MNEMONIC_POP || mn == ZYDIS_MNEMONIC_LEAVE) {
            std::uint64_t nva = d.next();
            bool ok = false;
            for (int step = 0; step < 8; ++step) {
                const DInsn* nd = fn.at(nva);
                if (nd == nullptr) break;
                if (nd->ins.meta.category == ZYDIS_CATEGORY_RET) {
                    ok = true;
                    break;
                }
                if (nd->ins.meta.category == ZYDIS_CATEGORY_COND_BR ||
                    nd->ins.meta.category == ZYDIS_CATEGORY_UNCOND_BR ||
                    nd->ins.meta.category == ZYDIS_CATEGORY_CALL) {
                    break;
                }
                for (std::uint8_t i = 0; i < nd->ins.operand_count; ++i) {
                    if (nd->ops[i].type == ZYDIS_OPERAND_TYPE_MEMORY &&
                        (Ctx::gpr_index(nd->ops[i].mem.base) == 5 || Ctx::gpr_index(nd->ops[i].mem.index) == 5)) {
                        ok = false;
                        step = 8;
                        break;
                    }
                }
                nva = nd->next();
            }
            if (ok) continue;
        }
        return false;
    }
    return true;
}

void update_stack_reg_mask(Ctx& c, const DInsn& d) {
    const auto mn = d.ins.mnemonic;
    if (d.ins.meta.category == ZYDIS_CATEGORY_CALL) {
        c.stack_reg_mask = static_cast<std::uint16_t>((1U << 4) | (c.rbp_is_frame_ptr ? (1U << 5) : 0U));
        return;
    }
    if (mn == ZYDIS_MNEMONIC_LEA && d.ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER && d.ops[0].size == 64 &&
        d.ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY && d.ops[1].mem.index == ZYDIS_REGISTER_NONE &&
        d.ops[1].mem.segment != ZYDIS_REGISTER_FS && d.ops[1].mem.segment != ZYDIS_REGISTER_GS) {
        const int dst = Ctx::gpr_index(d.ops[0].reg.value);
        const int src = Ctx::gpr_index(d.ops[1].mem.base);
        if (dst >= 0 && dst != 4) {
            if (src >= 0 && (c.stack_reg_mask & (1U << src)) != 0) {
                c.stack_reg_mask |= static_cast<std::uint16_t>(1U << dst);
            } else {
                c.stack_reg_mask &= static_cast<std::uint16_t>(~(1U << dst));
            }
        }
        return;
    }
    if (mn == ZYDIS_MNEMONIC_MOV && d.ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER && d.ops[0].size == 64 &&
        d.ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER && d.ops[1].size == 64) {
        const int dst = Ctx::gpr_index(d.ops[0].reg.value);
        const int src = Ctx::gpr_index(d.ops[1].reg.value);
        if (dst >= 0 && dst != 4) {
            if (src >= 0 && (c.stack_reg_mask & (1U << src)) != 0) {
                c.stack_reg_mask |= static_cast<std::uint16_t>(1U << dst);
            } else {
                c.stack_reg_mask &= static_cast<std::uint16_t>(~(1U << dst));
            }
        }
        return;
    }
    for (std::uint8_t i = 0; i < d.ins.operand_count; ++i) {
        const ZydisDecodedOperand& o = d.ops[i];
        if ((o.actions & ZYDIS_OPERAND_ACTION_MASK_WRITE) != 0 && o.type == ZYDIS_OPERAND_TYPE_REGISTER) {
            const int idx = Ctx::gpr_index(o.reg.value);
            if (idx >= 0 && idx != 4) {
                c.stack_reg_mask &= static_cast<std::uint16_t>(~(1U << idx));
            }
        }
    }
}

}  // namespace

bool is_fast_abi_eligible(const DFunction& fn, const ImageInfo& image) {
    if (!fn.error.empty() || fn.insns.empty()) return false;
    if (image.resumable.count(fn.entry) != 0) return false;
    if (!fn.call_targets.empty() || !fn.tail_targets.empty() ||
        !fn.unresolved_indirect_jumps.empty() || !fn.indirect_tail_jumps.empty()) {
        return false;
    }
    auto is_allowed_gpr = [](int idx) -> bool {
        return idx == 0 || idx == 1 || idx == 2 || idx == 4 || idx == 8 || idx == 9 || idx == 10 || idx == 11;
    };
    for (const auto& [va, d] : fn.insns) {
        (void)va;
        const auto cat = d.ins.meta.category;
        const auto mn = d.ins.mnemonic;
        if (cat == ZYDIS_CATEGORY_CALL || cat == ZYDIS_CATEGORY_FLAGOP || cat == ZYDIS_CATEGORY_X87_ALU ||
            cat == ZYDIS_CATEGORY_MMX || mn == ZYDIS_MNEMONIC_CPUID || mn == ZYDIS_MNEMONIC_LDMXCSR ||
            mn == ZYDIS_MNEMONIC_STMXCSR || mn == ZYDIS_MNEMONIC_VLDMXCSR || mn == ZYDIS_MNEMONIC_VSTMXCSR ||
            insn_is_trap(d) || insn_gated_ext(d) != nullptr) {
            return false;
        }
        if (cat == ZYDIS_CATEGORY_RET && d.ins.operand_count_visible > 0 && d.ops[0].imm.value.u != 0) {
            return false;
        }
        if (d.ins.cpu_flags != nullptr && d.ins.cpu_flags->tested != 0) {
            // Only allow flag tests when a flag-modifying instruction precedes in the same block;
            // conservative check: disallow flag reads in functions without a linear check below.
        }
        for (std::uint8_t i = 0; i < d.ins.operand_count; ++i) {
            const ZydisDecodedOperand& o = d.ops[i];
            if (o.type == ZYDIS_OPERAND_TYPE_MEMORY &&
                (o.mem.segment == ZYDIS_REGISTER_FS || o.mem.segment == ZYDIS_REGISTER_GS)) {
                return false;
            }
            if (o.type == ZYDIS_OPERAND_TYPE_REGISTER) {
                const int gi = r1::Ctx::gpr_index(o.reg.value);
                if (gi >= 0 && !is_allowed_gpr(gi)) return false;
                const int xi = r1::Ctx::xmm_index(o.reg.value);
                if (xi >= 4) return false;
            } else if (o.type == ZYDIS_OPERAND_TYPE_MEMORY) {
                const int bi = r1::Ctx::gpr_index(o.mem.base);
                if (bi >= 0 && !is_allowed_gpr(bi)) return false;
                const int ii = r1::Ctx::gpr_index(o.mem.index);
                if (ii >= 0 && !is_allowed_gpr(ii)) return false;
            }
        }
    }
    // Verify no block reads any CPU flag before writing it (RAX is passed in via kFastArgGprs).
    for (const std::uint64_t bstart : fn.block_starts) {
        std::uint64_t flags_defined = 0;
        std::uint64_t va = bstart;
        while (const DInsn* dp = fn.at(va)) {
            const DInsn& d = *dp;
            if (d.ins.cpu_flags != nullptr && (d.ins.cpu_flags->tested & ~flags_defined) != 0) {
                return false;
            }
            if (d.ins.cpu_flags != nullptr) {
                flags_defined |= (d.ins.cpu_flags->modified | d.ins.cpu_flags->set_0 | d.ins.cpu_flags->set_1);
                flags_defined &= ~d.ins.cpu_flags->undefined;
            }
            if (d.ins.meta.category == ZYDIS_CATEGORY_COND_BR || d.ins.meta.category == ZYDIS_CATEGORY_UNCOND_BR ||
                d.ins.meta.category == ZYDIS_CATEGORY_RET) {
                break;
            }
            va = d.next();
            if (fn.block_starts.count(va) != 0) break;
        }
    }
    return true;
}

namespace {

bool caller_reads_flags_after(const DFunction& fn, std::uint64_t ret_va) {
    std::uint64_t flags_defined = 0;
    std::uint64_t va = ret_va;
    for (int step = 0; step < 24; ++step) {
        const DInsn* dp = fn.at(va);
        if (dp == nullptr) return true;
        const DInsn& d = *dp;
        if (d.ins.cpu_flags != nullptr && (d.ins.cpu_flags->tested & ~flags_defined) != 0) return true;
        if (d.ins.meta.category == ZYDIS_CATEGORY_COND_BR) return true;
        if (d.ins.meta.category == ZYDIS_CATEGORY_RET) return false;
        if (d.ins.cpu_flags != nullptr) {
            flags_defined |= (d.ins.cpu_flags->modified | d.ins.cpu_flags->set_0 | d.ins.cpu_flags->set_1);
            flags_defined &= ~d.ins.cpu_flags->undefined;
        }
        if (d.ins.meta.category == ZYDIS_CATEGORY_UNCOND_BR) return true;
        va = d.next();
    }
    return true;
}

}  // namespace

bool R1Translator::Impl::lift_body(Impl& impl, const DFunction& fn, llvm::Function* f, bool fast_abi,
                                   std::string* why) {
    const auto t0 = std::chrono::steady_clock::now();
    llvm::BasicBlock* entry = llvm::BasicBlock::Create(*impl.context, "entry", f);
    Ctx c(*impl.module, f, fn, impl.image);
    c.is_fast_abi = fast_abi;
    c.fn_relaxed_tso =
        impl.image.relaxed_tso && impl.image.strict_tso_fns.count(fn.entry) == 0 && !function_has_sync(fn, impl.image);
    c.rbp_is_frame_ptr = function_uses_rbp_frame(fn);
    c.stack_reg_mask = static_cast<std::uint16_t>((1U << 4) | (c.rbp_is_frame_ptr ? (1U << 5) : 0U));
    c.B.SetInsertPoint(entry);
    for (int r = 0; r < 16; ++r) {
        c.gpr_slot[r] = c.B.CreateAlloca(c.B.getInt64Ty(), nullptr, "r" + std::to_string(r));
        c.xmm_slot[r] = c.B.CreateAlloca(c.v2i64(), nullptr, "x" + std::to_string(r));
    }
    for (int k = 0; k < r1::kFlagCount; ++k) {
        c.flag_slot[k] = c.B.CreateAlloca(c.B.getInt1Ty(), nullptr, "f" + std::to_string(k));
    }
    if (fast_abi) {
        for (int r = 0; r < 16; ++r) {
            c.B.CreateStore(c.B.getInt64(0), c.gpr_slot[r]);
            c.B.CreateStore(llvm::ConstantAggregateZero::get(c.v2i64()), c.xmm_slot[r]);
        }
        for (int k = 0; k < r1::kFlagCount; ++k) {
            c.B.CreateStore(c.B.getFalse(), c.flag_slot[k]);
        }
        for (int i = 0; i < 8; ++i) {
            c.B.CreateStore(f->getArg(1 + i), c.gpr_slot[kFastArgGprs[i]]);
        }
        for (int i = 0; i < 4; ++i) {
            c.B.CreateStore(f->getArg(9 + i), c.xmm_slot[i]);
        }
    } else {
        c.reload();
    }

    // A function that can resume (R3/R4) also starts a block at every call's return address:
    // `RtlRestoreContext` continues there (MSVC `longjmp` to a `setjmp` call site, or a context
    // that `RtlCaptureContext` took), so each one must be a case of the resume switch below.
    const bool resumable = !fast_abi && impl.image.resumable.count(fn.entry) != 0;
    std::set<std::uint64_t> starts = fn.block_starts;
    if (resumable) {
        for (const auto& kv : fn.insns) {
            const DInsn& d = kv.second;
            if (d.ins.meta.category == ZYDIS_CATEGORY_CALL && fn.at(d.next()) != nullptr) {
                starts.insert(d.next());
            }
        }
    }
    std::map<std::uint64_t, llvm::BasicBlock*> blocks;
    for (const std::uint64_t b : starts) {
        char name[24];
        std::snprintf(name, sizeof(name), "b%llx", static_cast<unsigned long long>(b));
        blocks[b] = llvm::BasicBlock::Create(*impl.context, name, f);
    }
    c.B.CreateBr(blocks.at(fn.entry));

    // Resume switch (R3): reload the restored context and continue at `rip`, which must be one of
    // this function's blocks (a catch continuation, an `__except` target, or a call's return
    // address).
    llvm::BasicBlock* resume = nullptr;
    if (resumable) {
        resume = llvm::BasicBlock::Create(*impl.context, "resume", f);
        llvm::BasicBlock* unknown = llvm::BasicBlock::Create(*impl.context, "resume_unknown", f);
        c.B.SetInsertPoint(resume);
        c.reload();
        llvm::Value* rip = c.B.CreateLoad(c.B.getInt64Ty(), c.state_ptr(cpu_offset::rip));
        llvm::SwitchInst* sw = c.B.CreateSwitch(rip, unknown, static_cast<unsigned>(blocks.size()));
        for (const auto& [va, bb] : blocks) {
            sw->addCase(c.B.getInt64(va), bb);
        }
        c.B.SetInsertPoint(unknown);
        c.emit_unsupported(fn.entry, "unwind continuation is not a block of this function");
    }

    std::string first_failure;
    auto fail = [&](std::uint64_t va, const std::string& what) {
        if (first_failure.empty()) {
            first_failure = what;
        }
        c.emit_unsupported(va, what);
    };
    // A tail jump leaves this frame: the call must not grow the host stack (R4: an interpreter's
    // handlers jump to each other for as long as it runs, `find_offset_table_entries`). A call to
    // a function of this module has this one's prototype, so it is `musttail` (guaranteed at every
    // optimization level); a call through a hook is marked `tail`, which -O1 and up honour.
    auto tail_call = [&](std::uint64_t target) {
        c.flush();
        if (impl.functions.count(target) != 0) {
            c.B.CreateCall(impl.function(target), {c.state})->setTailCallKind(llvm::CallInst::TCK_MustTail);
        } else {
            llvm::LoadInst* fp = c.B.CreateLoad(c.ptr(), impl.ext_call_slot_ptr(c.B, target));
            fp->setAlignment(llvm::Align(8));
            c.B.CreateCall(impl.guest_fn_ty(), fp, {c.state})->setTailCallKind(llvm::CallInst::TCK_MustTail);
        }
        c.B.CreateRetVoid();
    };

    for (const std::uint64_t start : starts) {
        c.B.SetInsertPoint(blocks.at(start));
        c.stack_reg_mask = static_cast<std::uint16_t>((1U << 4) | (c.rbp_is_frame_ptr ? (1U << 5) : 0U));
        std::uint64_t va = start;
        while (true) {
            const DInsn* dp = fn.at(va);
            if (dp == nullptr) {
                fail(va, "undecoded instruction");
                break;
            }
            const DInsn& d = *dp;
            c.last_address_stack = false;     // `lea` and atomics compute addresses without an access
            c.last_address_readonly = false;
            const auto cat = d.ins.meta.category;
            const auto mn = d.ins.mnemonic;
            bool terminated = false;

            if (cat == ZYDIS_CATEGORY_RET) {
                llvm::Value* rsp = c.gpr64(4);
                llvm::Value* ret = c.load(rsp, 64, r1::Mem::stack);
                std::uint64_t extra = d.ins.operand_count_visible > 0 ? d.ops[0].imm.value.u : 0;
                c.set_gpr64(4, c.B.CreateAdd(rsp, c.B.getInt64(8 + extra)));
                if (fast_abi) {
                    llvm::Value* agg = llvm::UndefValue::get(f->getReturnType());
                    for (int i = 0; i < 8; ++i) {
                        agg = c.B.CreateInsertValue(agg, c.gpr64(kFastRetGprs[i]), static_cast<unsigned>(i));
                    }
                    for (int i = 0; i < 4; ++i) {
                        agg = c.B.CreateInsertValue(agg, c.xmm_vec(i), static_cast<unsigned>(8 + i));
                    }
                    agg = c.B.CreateInsertValue(agg, ret, 12U);
                    c.B.CreateRet(agg);
                } else {
                    c.flush();
                    c.B.CreateStore(ret, c.state_ptr(cpu_offset::rip));
                    c.B.CreateRetVoid();
                }
                terminated = true;
            } else if (cat == ZYDIS_CATEGORY_COND_BR) {
                ZyanU64 target = 0;
                ZydisCalcAbsoluteAddress(&d.ins, &d.ops[0], d.va, &target);
                llvm::Value* cond = nullptr;
                if (mn == ZYDIS_MNEMONIC_JRCXZ || mn == ZYDIS_MNEMONIC_JECXZ || mn == ZYDIS_MNEMONIC_JCXZ) {
                    const unsigned w = mn == ZYDIS_MNEMONIC_JRCXZ ? 64 : (mn == ZYDIS_MNEMONIC_JECXZ ? 32 : 16);
                    cond = c.B.CreateICmpEQ(c.B.CreateTrunc(c.gpr64(1), c.ity(w)), llvm::ConstantInt::get(c.ity(w), 0));
                } else if (mn == ZYDIS_MNEMONIC_LOOP || mn == ZYDIS_MNEMONIC_LOOPE || mn == ZYDIS_MNEMONIC_LOOPNE) {
                    llvm::Value* rcx = c.B.CreateSub(c.gpr64(1), c.B.getInt64(1));
                    c.set_gpr64(1, rcx);
                    cond = c.B.CreateICmpNE(rcx, c.B.getInt64(0));
                    if (mn == ZYDIS_MNEMONIC_LOOPE) cond = c.B.CreateAnd(cond, c.flag(r1::ZF));
                    if (mn == ZYDIS_MNEMONIC_LOOPNE) cond = c.B.CreateAnd(cond, c.B.CreateNot(c.flag(r1::ZF)));
                } else {
                    cond = c.condition(mn);
                }
                if (cond == nullptr || blocks.count(target) == 0 || blocks.count(d.next()) == 0) {
                    fail(d.va, std::string("conditional branch ") + ZydisMnemonicGetString(mn));
                    break;
                }
                c.B.CreateCondBr(cond, blocks.at(target), blocks.at(d.next()));
                terminated = true;
            } else if (cat == ZYDIS_CATEGORY_UNCOND_BR) {
                if (d.ops[0].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
                    ZyanU64 target = 0;
                    ZydisCalcAbsoluteAddress(&d.ins, &d.ops[0], d.va, &target);
                    if (fn.tail_targets.count(target) != 0) {
                        tail_call(target);
                    } else if (blocks.count(target) != 0) {
                        c.B.CreateBr(blocks.at(target));
                    } else {
                        fail(d.va, "jmp to an unknown block");
                        break;
                    }
                } else {
                    const JumpTable* jt = nullptr;
                    for (const JumpTable& t : fn.jump_tables) {
                        if (t.jmp_va == d.va) {
                            jt = &t;
                        }
                    }
                    llvm::Value* target = c.B.CreateZExt(c.read(d, 0), c.B.getInt64Ty());
                    if (jt != nullptr) {
                        llvm::BasicBlock* dflt = llvm::BasicBlock::Create(c.C, "jt_default", f);
                        llvm::SwitchInst* sw = c.B.CreateSwitch(target, dflt);
                        std::set<std::uint64_t> seen;
                        for (const std::uint64_t t : jt->targets) {
                            if (seen.insert(t).second) {
                                sw->addCase(c.B.getInt64(t), blocks.at(t));
                            }
                        }
                        c.B.SetInsertPoint(dflt);
                        c.emit_unsupported(d.va, "jump table target outside the table");
                    } else {
                        // Indirect tail call (or a switch whose table was not found: the
                        // dispatcher aborts with its VA if the target is no function). A tail
                        // jump through an import slot (`jmp [rip + __imp_X]`) is an import call:
                        // the return address on the stack is the caller's, as the shim expects.
                        const ZydisDecodedOperand& o = d.ops[0];
                        const std::uint64_t slot = d.next() + static_cast<std::uint64_t>(o.mem.disp.value);
                        c.flush();
                        if (o.type == ZYDIS_OPERAND_TYPE_MEMORY && o.mem.base == ZYDIS_REGISTER_RIP &&
                            o.mem.index == ZYDIS_REGISTER_NONE && slot >= impl.image.iat_begin &&
                            slot < impl.image.iat_end) {
                            c.call_hook(offsetof(RuntimeHooks, import_call), {c.B.getInt64(slot)}, {c.B.getInt64Ty()})
                                ->setTailCall();
                        } else {
                            call_dispatch(c, target)->setTailCall();
                        }
                        c.B.CreateRetVoid();
                    }
                }
                terminated = true;
            } else if (cat == ZYDIS_CATEGORY_CALL) {
                const std::uint64_t ret_va = d.next();
                const ZydisDecodedOperand& o = d.ops[0];
                if (o.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
                    ZyanU64 target = 0;
                    ZydisCalcAbsoluteAddress(&d.ins, &o, d.va, &target);
                    const bool try_fast =
                        !fast_abi && !caller_reads_flags_after(fn, ret_va) &&
                        (impl.fast_functions.count(target) != 0 || impl.image.fast_abi_vas.count(target) != 0) &&
                        impl.inline_leaves.count(target) == 0;
                    if (try_fast) {
                        llvm::Value* site_rsp = c.gpr64(4);
                        llvm::Value* rsp = c.B.CreateSub(site_rsp, c.B.getInt64(8));
                        c.set_gpr64(4, rsp);
                        c.store(rsp, c.B.getInt64(ret_va), r1::Mem::stack);
                        std::vector<llvm::Value*> fargs{c.state};
                        for (int i = 0; i < 8; ++i) fargs.push_back(c.gpr64(kFastArgGprs[i]));
                        for (int i = 0; i < 4; ++i) fargs.push_back(c.xmm_vec(i));
                        llvm::CallInst* call = nullptr;
                        if (const auto fit = impl.fast_functions.find(target); fit != impl.fast_functions.end()) {
                            call = c.B.CreateCall(fit->second, fargs);
                        } else {
                            llvm::LoadInst* fp = c.B.CreateLoad(c.ptr(), impl.ext_fast_slot_ptr(c.B, target));
                            fp->setAlignment(llvm::Align(8));
                            call = c.B.CreateCall(impl.fast_fn_ty(), fp, fargs);
                        }
                        call->setCallingConv(llvm::CallingConv::Fast);
                        for (int i = 0; i < 8; ++i) {
                            c.set_gpr64(kFastRetGprs[i], c.B.CreateExtractValue(call, static_cast<unsigned>(i)));
                        }
                        for (int i = 0; i < 4; ++i) {
                            c.set_xmm(i, c.B.CreateExtractValue(call, static_cast<unsigned>(8 + i)));
                        }
                        llvm::Value* rip = c.B.CreateExtractValue(call, 12U);
                        llvm::BasicBlock* bad = llvm::BasicBlock::Create(c.C, "fast_bad_ret", f);
                        llvm::BasicBlock* ok = llvm::BasicBlock::Create(c.C, "fast_ret_ok", f);
                        c.B.CreateCondBr(c.B.CreateICmpEQ(rip, c.B.getInt64(ret_va)), ok, bad);
                        c.B.SetInsertPoint(bad);
                        c.flush();
                        c.B.CreateStore(rip, c.state_ptr(cpu_offset::rip));
                        c.call_hook(offsetof(RuntimeHooks, bad_return), {c.B.getInt64(ret_va)}, {c.B.getInt64Ty()});
                        c.B.CreateBr(ok);
                        c.B.SetInsertPoint(ok);
                    } else {
                        guest_call(c, ret_va, resume, [&] {
                            if (impl.functions.count(target) != 0) {
                                c.B.CreateCall(impl.function(target), {c.state});
                            } else if (const auto il = impl.inline_leaves.find(target); il != impl.inline_leaves.end()) {
                                c.B.CreateCall(il->second, {c.state});
                            } else {
                                llvm::LoadInst* fp = c.B.CreateLoad(c.ptr(), impl.ext_call_slot_ptr(c.B, target));
                                fp->setAlignment(llvm::Align(8));
                                c.B.CreateCall(impl.guest_fn_ty(), fp, {c.state});
                            }
                        });
                    }
                } else if (o.type == ZYDIS_OPERAND_TYPE_MEMORY && o.mem.base == ZYDIS_REGISTER_RIP &&
                           o.mem.index == ZYDIS_REGISTER_NONE) {
                    const std::uint64_t slot = d.next() + static_cast<std::uint64_t>(o.mem.disp.value);
                    if (slot >= impl.image.iat_begin && slot < impl.image.iat_end) {
                        guest_call(c, ret_va, resume, [&] {
                            c.call_hook(offsetof(RuntimeHooks, import_call), {c.B.getInt64(slot)}, {c.B.getInt64Ty()});
                        });
                    } else {
                        llvm::Value* target = c.read(d, 0);
                        guest_call(c, ret_va, resume, [&] { call_dispatch(c, target); });
                    }
                } else {
                    llvm::Value* target = c.B.CreateZExt(c.read(d, 0), c.B.getInt64Ty());
                    guest_call(c, ret_va, resume, [&] { call_dispatch(c, target); });
                }
            } else if (insn_is_trap(d)) {
                // A deliberate stop (MSVC puts `int3` after no-return calls; `int 0x29` is
                // __fastfail): reaching it at run time is a fault with its address, not missing work.
                ++impl.report.traps;
                c.emit_unsupported(d.va, ZydisMnemonicGetString(mn));
                break;
            } else if (const char* ext = insn_gated_ext(d)) {
                // Only reachable when `cpuid` advertises the feature, which the runtime's fixed
                // CPU profile does not (`Runtime::hook_cpuid`): a run-time fault if it is reached.
                impl.report.gated[ext]++;
                c.emit_unsupported(d.va, std::string("needs ") + ext + " (not advertised by cpuid)");
                break;
            } else if (mn == ZYDIS_MNEMONIC_CPUID) {
                c.flush();
                c.call_hook(offsetof(RuntimeHooks, cpuid), {}, {});
                c.reload();
            } else {
                Lifted r = r1::lift_int(c, d);
                if (r == Lifted::not_mine) {
                    r = r1::lift_sse(c, d);
                }
                if (r == Lifted::not_mine) {
                    r = r1::lift_string(c, d);
                }
                if (r != Lifted::yes) {
                    fail(d.va, r == Lifted::unsupported && !c.why.empty() ? c.why
                                                                          : std::string(ZydisMnemonicGetString(mn)));
                    break;
                }
            }
            if (terminated) {
                break;
            }
            update_stack_reg_mask(c, d);
            const std::uint64_t next = d.next();
            if (fn.tail_targets.count(next) != 0) {
                tail_call(next);
                break;
            }
            if (blocks.count(next) != 0) {
                c.B.CreateBr(blocks.at(next));
                break;
            }
            va = next;
        }
    }

    impl.lift_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::string verify;
    llvm::raw_string_ostream os(verify);
    if (llvm::verifyFunction(*f, &os)) {
        f->deleteBody();
        if (why != nullptr) {
            *why = "IR verification failed: " + os.str().substr(0, 300);
        }
        impl.report.unsupported["(IR verification)"]++;
        return false;
    }
    if (!first_failure.empty()) {
        impl.report.unsupported[first_failure]++;
        if (why != nullptr) {
            *why = first_failure;
        }
        return false;
    }
    return true;
}

void R1Translator::add_inline_leaf(const DFunction& fn) {
    if (impl_->functions.count(fn.entry) != 0 || impl_->inline_leaves.count(fn.entry) != 0) return;
    char name[40];
    std::snprintf(name, sizeof(name), "wse_inline_%llx", static_cast<unsigned long long>(fn.entry));
    auto* f = llvm::Function::Create(impl_->guest_fn_ty(), llvm::GlobalValue::InternalLinkage, name, *impl_->module);
    f->addFnAttr(llvm::Attribute::NoUnwind);
    f->addFnAttr(llvm::Attribute::InlineHint);
    std::string dummy;
    if (!Impl::lift_body(*impl_, fn, f, false, &dummy)) {
        f->eraseFromParent();
        return;
    }
    impl_->inline_leaves.emplace(fn.entry, f);
}

bool R1Translator::lift(const DFunction& fn, std::string* why) {
    ++impl_->report.attempted;
    llvm::Function* f = impl_->function(fn.entry);
    if (!f->empty()) {
        if (why != nullptr) {
            *why = "already lifted";
        }
        return false;
    }
    if (impl_->light.count(fn.entry) == 0 && is_fast_abi_eligible(fn, impl_->image)) {
        auto* fast_f = llvm::Function::Create(impl_->fast_fn_ty(), llvm::GlobalValue::ExternalLinkage,
                                              r1_fast_symbol(fn.entry), *impl_->module);
        fast_f->setVisibility(llvm::GlobalValue::ProtectedVisibility);
        fast_f->setDSOLocal(true);
        fast_f->setCallingConv(llvm::CallingConv::Fast);
        fast_f->addFnAttr(llvm::Attribute::NoUnwind);
        if (Impl::lift_body(*impl_, fn, fast_f, true, why)) {
            impl_->fast_functions.emplace(fn.entry, fast_f);
            // Emit thin CpuState* entry thunk `wse_fn_<va>` for indirect dispatch / host callbacks.
            llvm::IRBuilder<> b(*impl_->context);
            b.SetInsertPoint(llvm::BasicBlock::Create(*impl_->context, "thunk", f));
            llvm::Value* st = f->getArg(0);
            llvm::Type* i64 = b.getInt64Ty();
            llvm::Type* v2 = llvm::FixedVectorType::get(i64, 2);
            std::vector<llvm::Value*> args{st};
            for (int i = 0; i < 8; ++i) {
                args.push_back(b.CreateLoad(i64, b.CreateConstInBoundsGEP1_64(b.getInt8Ty(), st, cpu_offset::gpr + 8U * kFastArgGprs[i])));
            }
            for (int i = 0; i < 4; ++i) {
                llvm::LoadInst* xl = b.CreateLoad(v2, b.CreateConstInBoundsGEP1_64(b.getInt8Ty(), st, cpu_offset::xmm + 16U * i));
                xl->setAlignment(llvm::Align(16));
                args.push_back(xl);
            }
            llvm::CallInst* call = b.CreateCall(fast_f, args);
            call->setCallingConv(llvm::CallingConv::Fast);
            for (int i = 0; i < 8; ++i) {
                b.CreateStore(b.CreateExtractValue(call, static_cast<unsigned>(i)),
                              b.CreateConstInBoundsGEP1_64(b.getInt8Ty(), st, cpu_offset::gpr + 8U * kFastRetGprs[i]));
            }
            for (int i = 0; i < 4; ++i) {
                llvm::StoreInst* xs = b.CreateStore(
                    b.CreateExtractValue(call, static_cast<unsigned>(8 + i)),
                    b.CreateConstInBoundsGEP1_64(b.getInt8Ty(), st, cpu_offset::xmm + 16U * i));
                xs->setAlignment(llvm::Align(16));
            }
            b.CreateStore(b.CreateExtractValue(call, 12U),
                          b.CreateConstInBoundsGEP1_64(b.getInt8Ty(), st, cpu_offset::rip));
            b.CreateRetVoid();
            ++impl_->report.lifted;
            return true;
        }
        fast_f->eraseFromParent();
    }
    if (!Impl::lift_body(*impl_, fn, f, false, why)) {
        return false;
    }
    ++impl_->report.lifted;
    return true;
}

bool R1Translator::emit_object(int opt_level, std::vector<char>& object, TranslateTimings* timings,
                               std::string* error) {
    llvm::LLVMContext& ctx = *impl_->context;
    llvm::Module& m = *impl_->module;
    llvm::IRBuilder<> b(ctx);
    // Declared but not defined: forward to the dispatcher (another module, or a missing target).
    // Only defined functions go in the table, or a forwarder would dispatch to itself.
    std::set<std::uint64_t> defined;
    for (auto& [va, f] : impl_->functions) {
        if (!f->empty()) {
            defined.insert(va);
            continue;
        }
        b.SetInsertPoint(llvm::BasicBlock::Create(ctx, "fwd", f));
        llvm::Value* hooks = b.CreateLoad(b.getPtrTy(), b.CreateConstInBoundsGEP1_64(b.getInt8Ty(), f->getArg(0), cpu_offset::hooks));
        llvm::Value* fp = b.CreateLoad(b.getPtrTy(), b.CreateConstInBoundsGEP1_64(b.getInt8Ty(), hooks, offsetof(RuntimeHooks, dispatch)));
        auto* fty = llvm::FunctionType::get(b.getVoidTy(), {b.getPtrTy(), b.getInt64Ty()}, false);
        b.CreateCall(fty, fp, {f->getArg(0), b.getInt64(va)});
        b.CreateRetVoid();
    }
    // Dispatch table: sorted {va, fn} of everything in this module.
    std::vector<llvm::Constant*> rows;
    auto* row_ty = llvm::StructType::get(ctx, {b.getInt64Ty(), b.getPtrTy()});
    for (auto& [va, f] : impl_->functions) {
        if (defined.count(va) == 0) {
            continue;
        }
        rows.push_back(llvm::ConstantStruct::get(row_ty, {b.getInt64(va), f}));
    }
    auto* arr_ty = llvm::ArrayType::get(row_ty, rows.size());
    new llvm::GlobalVariable(m, arr_ty, true, llvm::GlobalValue::ExternalLinkage,
                             llvm::ConstantArray::get(arr_ty, rows), "wse_fn_table");
    new llvm::GlobalVariable(m, b.getInt64Ty(), true, llvm::GlobalValue::ExternalLinkage,
                             b.getInt64(rows.size()), "wse_fn_table_count");

    // Cross-batch direct call slots (`wse_ext_calls[]`): each entry starts pointing to a small
    // internal fallback thunk that calls `hooks->dispatch(state, va)` so an unpatched module still
    // works identically. Placed in writable `.data` (outside RELRO) for `Runtime::patch_external_calls`.
    const std::size_t ext_n = impl_->ext_call_vas.size();
    if (ext_n > 0) {
        std::vector<llvm::Constant*> ext_inits;
        std::vector<llvm::Constant*> ext_vas;
        ext_inits.reserve(ext_n);
        ext_vas.reserve(ext_n);
        auto* guest_ty = impl_->guest_fn_ty();
        auto* disp_ty = llvm::FunctionType::get(b.getVoidTy(), {b.getPtrTy(), b.getInt64Ty()}, false);
        for (const std::uint64_t va : impl_->ext_call_vas) {
            char stub_name[40];
            std::snprintf(stub_name, sizeof(stub_name), "wse_ext_fwd_%llx", static_cast<unsigned long long>(va));
            auto* stub = llvm::Function::Create(guest_ty, llvm::GlobalValue::InternalLinkage, stub_name, m);
            stub->addFnAttr(llvm::Attribute::NoUnwind);
            b.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", stub));
            llvm::Value* hooks = b.CreateLoad(
                b.getPtrTy(), b.CreateConstInBoundsGEP1_64(b.getInt8Ty(), stub->getArg(0), cpu_offset::hooks));
            llvm::Value* fp = b.CreateLoad(
                b.getPtrTy(), b.CreateConstInBoundsGEP1_64(b.getInt8Ty(), hooks, offsetof(RuntimeHooks, dispatch)));
            b.CreateCall(disp_ty, fp, {stub->getArg(0), b.getInt64(va)})->setTailCall();
            b.CreateRetVoid();
            ext_inits.push_back(stub);
            ext_vas.push_back(b.getInt64(va));
        }
        auto* ext_arr_ty = llvm::ArrayType::get(b.getPtrTy(), ext_n);
        auto* real_ext = new llvm::GlobalVariable(m, ext_arr_ty, false, llvm::GlobalValue::ExternalLinkage,
                                                  llvm::ConstantArray::get(ext_arr_ty, ext_inits), "wse_ext_calls");
        real_ext->setVisibility(llvm::GlobalValue::ProtectedVisibility);
        real_ext->setDSOLocal(true);
        real_ext->setSection(".data");
        if (impl_->ext_calls_gv != nullptr) {
            impl_->ext_calls_gv->replaceAllUsesWith(real_ext);
            impl_->ext_calls_gv->eraseFromParent();
            impl_->ext_calls_gv = nullptr;
        }
        auto* va_arr_ty = llvm::ArrayType::get(b.getInt64Ty(), ext_n);
        new llvm::GlobalVariable(m, va_arr_ty, true, llvm::GlobalValue::ExternalLinkage,
                                 llvm::ConstantArray::get(va_arr_ty, ext_vas), "wse_ext_call_vas");
    } else if (impl_->ext_calls_gv != nullptr) {
        impl_->ext_calls_gv->eraseFromParent();
        impl_->ext_calls_gv = nullptr;
    }
    new llvm::GlobalVariable(m, b.getInt64Ty(), true, llvm::GlobalValue::ExternalLinkage,
                             b.getInt64(ext_n), "wse_ext_call_count");

    // Fast-ABI table (`wse_fast_table`) and cross-batch fast call slots (`wse_ext_fast_calls[]`).
    {
        std::vector<llvm::Constant*> frows;
        frows.reserve(impl_->fast_functions.size());
        for (const auto& [va, ff] : impl_->fast_functions) {
            frows.push_back(llvm::ConstantStruct::get(row_ty, {b.getInt64(va), ff}));
        }
        auto* farr_ty = llvm::ArrayType::get(row_ty, frows.size());
        new llvm::GlobalVariable(m, farr_ty, true, llvm::GlobalValue::ExternalLinkage,
                                 llvm::ConstantArray::get(farr_ty, frows), "wse_fast_table");
        new llvm::GlobalVariable(m, b.getInt64Ty(), true, llvm::GlobalValue::ExternalLinkage,
                                 b.getInt64(frows.size()), "wse_fast_table_count");
    }
    const std::size_t ext_fast_n = impl_->ext_fast_vas.size();
    if (ext_fast_n > 0) {
        std::vector<llvm::Constant*> f_inits;
        std::vector<llvm::Constant*> f_vas;
        f_inits.reserve(ext_fast_n);
        f_vas.reserve(ext_fast_n);
        auto* fty = impl_->fast_fn_ty();
        auto* disp_ty = llvm::FunctionType::get(b.getVoidTy(), {b.getPtrTy(), b.getInt64Ty()}, false);
        llvm::Type* i64 = b.getInt64Ty();
        llvm::Type* v2 = llvm::FixedVectorType::get(i64, 2);
        for (const std::uint64_t va : impl_->ext_fast_vas) {
            char stub_name[48];
            std::snprintf(stub_name, sizeof(stub_name), "wse_ext_fast_fwd_%llx", static_cast<unsigned long long>(va));
            auto* stub = llvm::Function::Create(fty, llvm::GlobalValue::InternalLinkage, stub_name, m);
            stub->setCallingConv(llvm::CallingConv::Fast);
            stub->addFnAttr(llvm::Attribute::NoUnwind);
            b.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", stub));
            llvm::Value* st = stub->getArg(0);
            for (int i = 0; i < 8; ++i) {
                b.CreateStore(stub->getArg(1 + i),
                              b.CreateConstInBoundsGEP1_64(b.getInt8Ty(), st, cpu_offset::gpr + 8U * kFastArgGprs[i]));
            }
            for (int i = 0; i < 4; ++i) {
                llvm::StoreInst* xs = b.CreateStore(
                    stub->getArg(9 + i),
                    b.CreateConstInBoundsGEP1_64(b.getInt8Ty(), st, cpu_offset::xmm + 16U * i));
                xs->setAlignment(llvm::Align(16));
            }
            llvm::Value* hooks = b.CreateLoad(
                b.getPtrTy(), b.CreateConstInBoundsGEP1_64(b.getInt8Ty(), st, cpu_offset::hooks));
            llvm::Value* fp = b.CreateLoad(
                b.getPtrTy(), b.CreateConstInBoundsGEP1_64(b.getInt8Ty(), hooks, offsetof(RuntimeHooks, dispatch)));
            b.CreateCall(disp_ty, fp, {st, b.getInt64(va)});
            llvm::Value* agg = llvm::UndefValue::get(impl_->fast_ret_ty());
            for (int i = 0; i < 8; ++i) {
                agg = b.CreateInsertValue(
                    agg, b.CreateLoad(i64, b.CreateConstInBoundsGEP1_64(b.getInt8Ty(), st, cpu_offset::gpr + 8U * kFastRetGprs[i])),
                    static_cast<unsigned>(i));
            }
            for (int i = 0; i < 4; ++i) {
                llvm::LoadInst* xl = b.CreateLoad(
                    v2, b.CreateConstInBoundsGEP1_64(b.getInt8Ty(), st, cpu_offset::xmm + 16U * i));
                xl->setAlignment(llvm::Align(16));
                agg = b.CreateInsertValue(agg, xl, static_cast<unsigned>(8 + i));
            }
            agg = b.CreateInsertValue(
                agg, b.CreateLoad(i64, b.CreateConstInBoundsGEP1_64(b.getInt8Ty(), st, cpu_offset::rip)), 12U);
            b.CreateRet(agg);
            f_inits.push_back(stub);
            f_vas.push_back(b.getInt64(va));
        }
        auto* farr_ty = llvm::ArrayType::get(b.getPtrTy(), ext_fast_n);
        auto* real_fext = new llvm::GlobalVariable(m, farr_ty, false, llvm::GlobalValue::ExternalLinkage,
                                                   llvm::ConstantArray::get(farr_ty, f_inits), "wse_ext_fast_calls");
        real_fext->setVisibility(llvm::GlobalValue::ProtectedVisibility);
        real_fext->setDSOLocal(true);
        real_fext->setSection(".data");
        if (impl_->ext_fast_gv != nullptr) {
            impl_->ext_fast_gv->replaceAllUsesWith(real_fext);
            impl_->ext_fast_gv->eraseFromParent();
            impl_->ext_fast_gv = nullptr;
        }
        auto* va_arr_ty = llvm::ArrayType::get(b.getInt64Ty(), ext_fast_n);
        new llvm::GlobalVariable(m, va_arr_ty, true, llvm::GlobalValue::ExternalLinkage,
                                 llvm::ConstantArray::get(va_arr_ty, f_vas), "wse_ext_fast_call_vas");
    } else if (impl_->ext_fast_gv != nullptr) {
        impl_->ext_fast_gv->eraseFromParent();
        impl_->ext_fast_gv = nullptr;
    }
    new llvm::GlobalVariable(m, b.getInt64Ty(), true, llvm::GlobalValue::ExternalLinkage,
                             b.getInt64(ext_fast_n), "wse_ext_fast_call_count");

    const llvm_common::TargetSetup t = llvm_common::make_target(opt_level);
    if (!t.tm) {
        if (error != nullptr) {
            *error = t.error;
        }
        return false;
    }
    for (const std::uint64_t va : impl_->light) {
        const auto it = impl_->functions.find(va);
        if (it != impl_->functions.end() && !it->second->empty()) {
            it->second->addFnAttr(llvm::Attribute::NoInline);
            it->second->addFnAttr(llvm::Attribute::OptimizeNone);
        }
    }
    const double lift_ms = impl_->lift_ms;
    impl_->total_ms += lift_ms;
    const bool ok = llvm_common::optimize_and_emit(m, *t.tm, opt_level, object, timings, error);
    if (timings != nullptr) {
        timings->lift_ms = lift_ms;
    }
    impl_->reset();
    return ok;
}

}  // namespace wse::recomp
