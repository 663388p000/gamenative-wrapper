/*
 * SPIR-V analysis / lowering for the feature emulation layer.
 * See wrapper_spirv_emu.h. Everything here works on a parsed instruction list
 * and either produces a complete new module or reports failure.
 */

#include "wrapper_spirv_emu.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

enum : uint32_t {
   SpvMagic = 0x07230203,
   OpMemberName_ = 6, OpEntryPoint_ = 15, OpCapability_ = 17,
   OpTypeBool_ = 20, OpTypeInt_ = 21, OpTypeFloat_ = 22, OpTypeVector_ = 23,
   OpTypeMatrix_ = 24, OpTypeArray_ = 28, OpTypeStruct_ = 30, OpTypePointer_ = 32,
   OpConstant_ = 43,
   OpFunction_ = 54, OpFunctionCall_ = 57,
   OpVariable_ = 59, OpLoad_ = 61, OpStore_ = 62, OpCopyMemory_ = 63,
   OpCopyMemorySized_ = 64, OpAccessChain_ = 65, OpInBoundsAccessChain_ = 66,
   OpDecorate_ = 71, OpMemberDecorate_ = 72,
   OpLogicalOr_ = 166, OpFOrdLessThan_ = 184,
   OpSelectionMerge_ = 247, OpLabel_ = 248, OpBranchConditional_ = 250, OpKill_ = 252,
};
enum : uint32_t { DecBuiltIn = 11, DecLocation = 30 };
enum : uint32_t { BiClipDistance = 3, BiCullDistance = 4, BiLayer = 9, BiViewportIndex = 10 };
enum : uint32_t { StInput = 1, StOutput = 3 };
enum : uint32_t { CapClipDistance = 32, CapCullDistance = 33 };

struct Inst {
   uint32_t op = 0;
   std::vector<uint32_t> w; /* operands, without the first word */
   Inst() {}
   Inst(uint32_t o, std::initializer_list<uint32_t> l) : op(o), w(l) {}
};

static bool
result_id(const Inst &i, uint32_t &id)
{
   uint32_t op = i.op;
   size_t pos;
   if (op >= 19 && op <= 38)
      pos = 0;               /* types */
   else if (op == OpLabel_)
      pos = 0;
   else if ((op >= 41 && op <= 46) || (op >= 48 && op <= 52) || op == OpFunction_ ||
            op == 55 || op == OpVariable_ || op == OpLoad_ ||
            op == OpAccessChain_ || op == OpInBoundsAccessChain_)
      pos = 1;               /* constants, functions, variables, loads, chains */
   else
      return false;
   if (i.w.size() <= pos)
      return false;
   id = i.w[pos];
   return true;
}

struct Entry {
   size_t inst;
   uint32_t model, func;
   size_t iface; /* index into w[] of the first interface id */
};

struct VarInfo {
   uint32_t id, storage, pointee;
};

struct Module {
   uint32_t hdr[5];
   std::vector<Inst> v;
   std::unordered_map<uint32_t, size_t> defs;

   bool parse(const uint32_t *code, size_t words)
   {
      if (words < 5 || code[0] != SpvMagic)
         return false;
      memcpy(hdr, code, sizeof(hdr));
      v.clear();
      size_t i = 5;
      while (i < words) {
         uint32_t wc = code[i] >> 16;
         uint32_t op = code[i] & 0xffff;
         if (wc == 0 || i + wc > words)
            return false;
         Inst in;
         in.op = op;
         in.w.assign(code + i + 1, code + i + wc);
         v.push_back(std::move(in));
         i += wc;
      }
      index();
      return true;
   }

   void serialize(std::vector<uint32_t> &out) const
   {
      out.assign(hdr, hdr + 5);
      for (const Inst &in : v) {
         out.push_back(((uint32_t)(in.w.size() + 1) << 16) | in.op);
         out.insert(out.end(), in.w.begin(), in.w.end());
      }
   }

   uint32_t new_id() { return hdr[3]++; }

   void index()
   {
      defs.clear();
      for (size_t k = 0; k < v.size(); k++) {
         uint32_t id;
         if (result_id(v[k], id))
            defs[id] = k;
      }
   }

   const Inst *def(uint32_t id) const
   {
      auto it = defs.find(id);
      return it == defs.end() ? nullptr : &v[it->second];
   }

   bool decor(uint32_t id, uint32_t deco, uint32_t *lit = nullptr) const
   {
      for (const Inst &in : v) {
         if (in.op == OpDecorate_ && in.w.size() >= 2 && in.w[0] == id && in.w[1] == deco) {
            if (lit)
               *lit = in.w.size() > 2 ? in.w[2] : 0;
            return true;
         }
      }
      return false;
   }

   bool member_decor(uint32_t st, uint32_t member, uint32_t deco, uint32_t *lit = nullptr) const
   {
      for (const Inst &in : v) {
         if (in.op == OpMemberDecorate_ && in.w.size() >= 3 && in.w[0] == st &&
             in.w[1] == member && in.w[2] == deco) {
            if (lit)
               *lit = in.w.size() > 3 ? in.w[3] : 0;
            return true;
         }
      }
      return false;
   }

   bool const_u32(uint32_t id, uint32_t &val) const
   {
      const Inst *d = def(id);
      if (!d || d->op != OpConstant_ || d->w.size() < 3)
         return false;
      val = d->w[2];
      return true;
   }

   /* OpTypeArray -> element type + length */
   bool arr_info(uint32_t type, uint32_t &elem, uint32_t &n) const
   {
      const Inst *d = def(type);
      if (!d || d->op != OpTypeArray_ || d->w.size() < 3)
         return false;
      elem = d->w[1];
      return const_u32(d->w[2], n);
   }

   /* locations a type occupies in a stage interface; WSPV_SLOTS_UNKNOWN if unsure */
   uint32_t slots(uint32_t type) const
   {
      const Inst *d = def(type);
      if (!d)
         return WSPV_SLOTS_UNKNOWN;
      switch (d->op) {
      case OpTypeBool_:
      case OpTypeInt_:
      case OpTypeFloat_:
         return 1;
      case OpTypeVector_: {
         const Inst *e = def(d->w[1]);
         if (!e || d->w.size() < 3)
            return WSPV_SLOTS_UNKNOWN;
         uint32_t width = (e->op == OpTypeInt_ || e->op == OpTypeFloat_) ? e->w[1] : 32;
         return (width == 64 && d->w[2] >= 3) ? 2 : 1;
      }
      case OpTypeMatrix_: {
         uint32_t c = slots(d->w[1]);
         return c == WSPV_SLOTS_UNKNOWN ? c : c * d->w[2];
      }
      case OpTypeArray_: {
         uint32_t e, n;
         if (!arr_info(type, e, n))
            return WSPV_SLOTS_UNKNOWN;
         uint32_t s = slots(e);
         return s == WSPV_SLOTS_UNKNOWN ? s : s * n;
      }
      default:
         return WSPV_SLOTS_UNKNOWN;
      }
   }

   bool find_entry(uint32_t model, const char *name, Entry &e) const
   {
      for (size_t k = 0; k < v.size(); k++) {
         const Inst &i = v[k];
         if (i.op != OpEntryPoint_ || i.w.size() < 3 || i.w[0] != model)
            continue;
         std::string s;
         size_t j = 2;
         bool done = false;
         for (; j < i.w.size() && !done; j++) {
            for (int b = 0; b < 4; b++) {
               char c = (char)((i.w[j] >> (8 * b)) & 0xff);
               if (!c) {
                  done = true;
                  break;
               }
               s.push_back(c);
            }
         }
         if (!done || s != name)
            continue;
         e.inst = k;
         e.model = i.w[0];
         e.func = i.w[1];
         e.iface = j;
         return true;
      }
      return false;
   }

   bool var_info(uint32_t id, VarInfo &vi) const
   {
      const Inst *d = def(id);
      if (!d || d->op != OpVariable_ || d->w.size() < 3)
         return false;
      const Inst *pt = def(d->w[0]);
      if (!pt || pt->op != OpTypePointer_ || pt->w.size() < 3)
         return false;
      vi.id = id;
      vi.storage = d->w[2];
      vi.pointee = pt->w[2];
      return true;
   }

   size_t first_function() const
   {
      for (size_t k = 0; k < v.size(); k++)
         if (v[k].op == OpFunction_)
            return k;
      return v.size();
   }

   /* end of the annotation section = just before the first type/constant/global */
   size_t annotation_end() const
   {
      for (size_t k = 0; k < v.size(); k++) {
         uint32_t op = v[k].op;
         if ((op >= 19 && op <= 39) || (op >= 41 && op <= 52) || op == OpVariable_ || op == OpFunction_)
            return k;
      }
      return v.size();
   }

   bool has_builtin_decoration(uint32_t builtin) const
   {
      for (const Inst &in : v) {
         if (in.op == OpDecorate_ && in.w.size() >= 3 && in.w[1] == DecBuiltIn && in.w[2] == builtin)
            return true;
         if (in.op == OpMemberDecorate_ && in.w.size() >= 4 && in.w[2] == DecBuiltIn && in.w[3] == builtin)
            return true;
      }
      return false;
   }

   void drop_capability_if_unused(uint32_t cap, uint32_t builtin)
   {
      if (has_builtin_decoration(builtin))
         return;
      for (size_t k = 0; k < v.size(); k++) {
         if (v[k].op == OpCapability_ && !v[k].w.empty() && v[k].w[0] == cap) {
            v.erase(v.begin() + k);
            break;
         }
      }
      index();
   }
};

/* New global instructions (types, constants, variables) collected while a
 * lowering runs; committed right before the first function. */
struct Pending {
   Module &m;
   std::vector<Inst> list;

   explicit Pending(Module &mm) : m(mm) {}

   uint32_t find(uint32_t op, std::initializer_list<uint32_t> tail) const
   {
      auto match = [&](const Inst &in) {
         if (in.op != op || in.w.size() != tail.size() + 1)
            return false;
         size_t j = 1;
         for (uint32_t t : tail)
            if (in.w[j++] != t)
               return false;
         return true;
      };
      for (const Inst &in : m.v)
         if (match(in))
            return in.w[0];
      for (const Inst &in : list)
         if (match(in))
            return in.w[0];
      return 0;
   }

   /* constants have their result id at w[1] */
   uint32_t find_const(uint32_t type, uint32_t bits) const
   {
      for (const Inst &in : m.v)
         if (in.op == OpConstant_ && in.w.size() == 3 && in.w[0] == type && in.w[2] == bits)
            return in.w[1];
      for (const Inst &in : list)
         if (in.op == OpConstant_ && in.w.size() == 3 && in.w[0] == type && in.w[2] == bits)
            return in.w[1];
      return 0;
   }

   uint32_t type_f32()
   {
      uint32_t id = find(OpTypeFloat_, {32});
      if (!id) {
         id = m.new_id();
         list.push_back(Inst(OpTypeFloat_, {id, 32}));
      }
      return id;
   }
   uint32_t type_u32()
   {
      uint32_t id = find(OpTypeInt_, {32, 0});
      if (!id) {
         id = m.new_id();
         list.push_back(Inst(OpTypeInt_, {id, 32, 0}));
      }
      return id;
   }
   uint32_t type_bool()
   {
      uint32_t id = find(OpTypeBool_, {});
      if (!id) {
         id = m.new_id();
         list.push_back(Inst(OpTypeBool_, {id}));
      }
      return id;
   }
   uint32_t ptr(uint32_t storage, uint32_t pointee)
   {
      uint32_t id = find(OpTypePointer_, {storage, pointee});
      if (!id) {
         id = m.new_id();
         list.push_back(Inst(OpTypePointer_, {id, storage, pointee}));
      }
      return id;
   }
   uint32_t constant(uint32_t type, uint32_t bits)
   {
      uint32_t id = find_const(type, bits);
      if (!id) {
         id = m.new_id();
         list.push_back(Inst(OpConstant_, {type, id, bits}));
      }
      return id;
   }
   uint32_t const_f32(float f)
   {
      uint32_t bits;
      memcpy(&bits, &f, 4);
      return constant(type_f32(), bits);
   }
   uint32_t array(uint32_t elem, uint32_t n)
   {
      auto length = [&](uint32_t len_id, uint32_t &val) {
         for (const Inst &in : m.v)
            if (in.op == OpConstant_ && in.w.size() >= 3 && in.w[1] == len_id) {
               val = in.w[2];
               return true;
            }
         for (const Inst &in : list)
            if (in.op == OpConstant_ && in.w.size() >= 3 && in.w[1] == len_id) {
               val = in.w[2];
               return true;
            }
         return false;
      };
      for (const std::vector<Inst> *src : {&m.v, &list}) {
         for (const Inst &in : *src) {
            uint32_t val;
            if (in.op == OpTypeArray_ && in.w.size() == 3 && in.w[1] == elem &&
                length(in.w[2], val) && val == n)
               return in.w[0];
         }
      }
      uint32_t len = constant(type_u32(), n);
      uint32_t id = m.new_id();
      list.push_back(Inst(OpTypeArray_, {id, elem, len}));
      return id;
   }

   void commit()
   {
      size_t at = m.first_function();
      m.v.insert(m.v.begin() + at, list.begin(), list.end());
      list.clear();
      m.index();
   }
};

static bool
emit(const Module &m, uint32_t **out, size_t *out_size_bytes)
{
   std::vector<uint32_t> words;
   m.serialize(words);
   uint32_t *p = (uint32_t *)malloc(words.size() * sizeof(uint32_t));
   if (!p)
      return false;
   memcpy(p, words.data(), words.size() * sizeof(uint32_t));
   *out = p;
   *out_size_bytes = words.size() * sizeof(uint32_t);
   return true;
}

/* ---------------------------------------------------------------- analysis */

static void
classify(const Module &m, wrapper_spv_stage_info *out, uint32_t builtin, uint32_t type)
{
   uint32_t elem, n;
   switch (builtin) {
   case BiClipDistance:
      out->has_clip = true;
      if (m.arr_info(type, elem, n))
         out->clip_count = n;
      break;
   case BiCullDistance:
      out->has_cull = true;
      if (m.arr_info(type, elem, n))
         out->cull_count = n;
      break;
   case BiViewportIndex:
      out->has_viewport_index = true;
      break;
   case BiLayer:
      out->has_layer = true;
      break;
   default:
      break;
   }
}

static bool
analyze_module(const Module &m, uint32_t model, const char *name, wrapper_spv_stage_info *out)
{
   memset(out, 0, sizeof(*out));
   Entry e;
   if (!m.find_entry(model, name, e))
      return true;
   out->found = true;

   const uint32_t storage = model == WSPV_FRAGMENT ? StInput : StOutput;
   const Inst &ep = m.v[e.inst];
   uint32_t end = 0;
   bool unknown = false;

   for (size_t k = e.iface; k < ep.w.size(); k++) {
      VarInfo vi;
      if (!m.var_info(ep.w[k], vi) || vi.storage != storage)
         continue;

      uint32_t bi;
      if (m.decor(vi.id, DecBuiltIn, &bi)) {
         classify(m, out, bi, vi.pointee);
         continue;
      }

      const Inst *t = m.def(vi.pointee);
      if (t && t->op == OpTypeStruct_ && !t->w.empty()) {
         for (uint32_t mi = 0; mi + 1 < t->w.size(); mi++) {
            uint32_t loc, mtype = t->w[1 + mi];
            if (m.member_decor(t->w[0], mi, DecBuiltIn, &bi)) {
               classify(m, out, bi, mtype);
            } else if (m.member_decor(t->w[0], mi, DecLocation, &loc)) {
               uint32_t s = m.slots(mtype);
               if (s == WSPV_SLOTS_UNKNOWN)
                  unknown = true;
               else
                  end = std::max(end, loc + s);
            } else {
               unknown = true;
            }
         }
         continue;
      }

      uint32_t loc;
      if (!m.decor(vi.id, DecLocation, &loc)) {
         unknown = true;
         continue;
      }
      uint32_t s = m.slots(vi.pointee);
      if (s == WSPV_SLOTS_UNKNOWN)
         unknown = true;
      else
         end = std::max(end, loc + s);
   }

   out->slots_end = unknown ? WSPV_SLOTS_UNKNOWN : end;
   return true;
}

/* ------------------------------------------------------------ VS lowering */

/*
 * Moves a builtin float array (ClipDistance/CullDistance) written by the
 * vertex shader to a plain Output float[N] variable at `loc`.
 *
 *  - standalone builtin variable: the decoration just changes from BuiltIn to
 *    Location;
 *  - member of the gl_PerVertex-style block: the member is removed from the
 *    struct (members after it are renumbered, member decorations / names and
 *    the constant member indices of every access chain follow) and each store
 *    into it becomes a store into the new variable.
 */
static bool
lower_builtin_vs(Module &m, const char *name, uint32_t builtin, uint32_t cap, uint32_t loc)
{
   Entry e;
   if (!m.find_entry(WSPV_VERTEX, name, e))
      return false;

   uint32_t V = 0, S = 0, A = 0;
   int k = -1; /* -2: standalone variable */
   {
      const Inst &ep = m.v[e.inst];
      for (size_t j = e.iface; j < ep.w.size() && !V; j++) {
         VarInfo vi;
         if (!m.var_info(ep.w[j], vi) || vi.storage != StOutput)
            continue;
         uint32_t bi;
         if (m.decor(vi.id, DecBuiltIn, &bi)) {
            if (bi == builtin) {
               V = vi.id;
               A = vi.pointee;
               k = -2;
            }
            continue;
         }
         const Inst *t = m.def(vi.pointee);
         if (t && t->op == OpTypeStruct_) {
            for (uint32_t mi = 0; mi + 1 < t->w.size(); mi++) {
               if (m.member_decor(t->w[0], mi, DecBuiltIn, &bi) && bi == builtin) {
                  V = vi.id;
                  S = t->w[0];
                  A = t->w[1 + mi];
                  k = (int)mi;
                  break;
               }
            }
         }
      }
   }
   if (!V)
      return false;

   uint32_t elemT, n;
   if (!m.arr_info(A, elemT, n) || n == 0)
      return false;

   if (k == -2) {
      for (Inst &in : m.v) {
         if (in.op == OpDecorate_ && in.w.size() >= 3 && in.w[0] == V &&
             in.w[1] == DecBuiltIn && in.w[2] == builtin) {
            in.w = {V, DecLocation, loc};
            m.drop_capability_if_unused(cap, builtin);
            m.index();
            return true;
         }
      }
      return false;
   }

   /* ---------- block member form ---------- */
   struct Chain {
      uint32_t base;
      std::vector<uint32_t> idx;
   };
   std::unordered_map<uint32_t, Chain> chains;
   for (const Inst &in : m.v) {
      if ((in.op == OpAccessChain_ || in.op == OpInBoundsAccessChain_) && in.w.size() >= 4)
         chains[in.w[1]] = Chain{in.w[2], std::vector<uint32_t>(in.w.begin() + 3, in.w.end())};
   }
   auto resolve = [&](uint32_t p, std::vector<uint32_t> &idx) {
      std::vector<const Chain *> stack;
      for (size_t guard = 0; guard < 64; guard++) {
         auto it = chains.find(p);
         if (it == chains.end())
            break;
         stack.push_back(&it->second);
         p = it->second.base;
      }
      idx.clear();
      for (auto it = stack.rbegin(); it != stack.rend(); ++it)
         idx.insert(idx.end(), (*it)->idx.begin(), (*it)->idx.end());
      return p;
   };

   Pending P(m);
   const uint32_t SV = m.new_id();
   const uint32_t PA = P.ptr(StOutput, A);
   uint32_t PF = 0;

   std::vector<char> erase(m.v.size(), 0);
   std::unordered_map<size_t, std::vector<Inst>> replace;
   std::vector<std::pair<size_t, uint32_t>> patch; /* direct chain on V: member index > k */

   for (size_t i = 0; i < m.v.size(); i++) {
      const Inst &in = m.v[i];
      std::vector<uint32_t> idx;
      uint32_t mi;

      if ((in.op == OpAccessChain_ || in.op == OpInBoundsAccessChain_) && in.w.size() >= 4) {
         if (resolve(in.w[1], idx) != V || idx.empty())
            continue;
         if (!m.const_u32(idx[0], mi))
            return false;
         if (mi == (uint32_t)k)
            erase[i] = 1;
         else if (mi > (uint32_t)k && in.w[2] == V)
            patch.push_back({i, mi});
      } else if (in.op == OpStore_ && in.w.size() >= 2) {
         if (resolve(in.w[0], idx) != V)
            continue;
         if (idx.empty() || !m.const_u32(idx[0], mi))
            return false;
         if (mi != (uint32_t)k)
            continue;
         erase[i] = 1;
         if (idx.size() == 1) {
            replace[i] = {Inst(OpStore_, {SV, in.w[1]})};
         } else if (idx.size() == 2) {
            if (!PF)
               PF = P.ptr(StOutput, elemT);
            uint32_t t = m.new_id();
            replace[i] = {Inst(OpAccessChain_, {PF, t, SV, idx[1]}), Inst(OpStore_, {t, in.w[1]})};
         } else {
            return false;
         }
      } else if (in.op == OpLoad_ && in.w.size() >= 3) {
         if (resolve(in.w[2], idx) != V)
            continue;
         if (idx.empty() || !m.const_u32(idx[0], mi) || mi == (uint32_t)k)
            return false; /* the shader reads the builtin back */
      } else if (in.op == OpCopyMemory_ || in.op == OpCopyMemorySized_) {
         for (size_t a = 0; a < 2 && a < in.w.size(); a++)
            if (resolve(in.w[a], idx) == V)
               return false;
      } else if (in.op == OpFunctionCall_) {
         for (size_t a = 3; a < in.w.size(); a++)
            if (resolve(in.w[a], idx) == V)
               return false;
      }
   }

   /* shift the constant member index of chains that look at members behind k */
   for (auto &pr : patch) {
      Inst &in = m.v[pr.first];
      const Inst *c = m.def(in.w[3]);
      if (!c || c->op != OpConstant_ || c->w.size() < 3)
         return false;
      in.w[3] = P.constant(c->w[0], pr.second - 1);
   }

   std::vector<Inst> nv;
   nv.reserve(m.v.size() + 8);
   for (size_t i = 0; i < m.v.size(); i++) {
      if (!erase[i]) {
         nv.push_back(std::move(m.v[i]));
      } else {
         auto it = replace.find(i);
         if (it != replace.end())
            for (Inst &r : it->second)
               nv.push_back(std::move(r));
      }
   }
   m.v.swap(nv);

   /* struct: drop member k, renumber the rest */
   for (Inst &in : m.v) {
      if (in.op == OpTypeStruct_ && !in.w.empty() && in.w[0] == S) {
         if (in.w.size() < (size_t)k + 2)
            return false;
         in.w.erase(in.w.begin() + 1 + k);
      }
   }
   std::vector<Inst> nv2;
   nv2.reserve(m.v.size());
   for (Inst &in : m.v) {
      if ((in.op == OpMemberDecorate_ || in.op == OpMemberName_) && in.w.size() >= 2 && in.w[0] == S) {
         if (in.w[1] == (uint32_t)k)
            continue;
         if (in.w[1] > (uint32_t)k)
            in.w[1]--;
      }
      nv2.push_back(std::move(in));
   }
   m.v.swap(nv2);

   /* new output variable + decoration + interface entry */
   P.list.push_back(Inst(OpVariable_, {PA, SV, StOutput}));
   m.index();
   P.commit();
   m.v.insert(m.v.begin() + m.annotation_end(), Inst(OpDecorate_, {SV, DecLocation, loc}));
   m.index();

   if (!m.find_entry(WSPV_VERTEX, name, e))
      return false;
   m.v[e.inst].w.push_back(SV);

   m.drop_capability_if_unused(cap, builtin);
   return true;
}

/* ------------------------------------------------------------ FS lowering */

static bool
lower_clip_fs_module(Module &m, const char *name, uint32_t count, uint32_t loc)
{
   Entry e;
   if (!m.find_entry(WSPV_FRAGMENT, name, e) || count == 0)
      return false;

   Pending P(m);
   uint32_t SV = 0;
   bool added_var = false;

   {
      const Inst &ep = m.v[e.inst];
      for (size_t j = e.iface; j < ep.w.size(); j++) {
         VarInfo vi;
         if (!m.var_info(ep.w[j], vi) || vi.storage != StInput)
            continue;
         uint32_t bi;
         if (m.decor(vi.id, DecBuiltIn, &bi)) {
            if (bi == BiClipDistance) {
               uint32_t el, n;
               if (!m.arr_info(vi.pointee, el, n) || n != count)
                  return false;
               SV = vi.id;
            }
            continue;
         }
         const Inst *t = m.def(vi.pointee);
         if (t && t->op == OpTypeStruct_) {
            for (uint32_t mi = 0; mi + 1 < t->w.size(); mi++)
               if (m.member_decor(t->w[0], mi, DecBuiltIn, &bi) && bi == BiClipDistance)
                  return false; /* clip distance read through an input block */
         }
      }
   }

   const uint32_t F32 = P.type_f32();
   const uint32_t U32 = P.type_u32();
   const uint32_t BOOL = P.type_bool();
   const uint32_t PIF = P.ptr(StInput, F32);
   const uint32_t ZERO = P.const_f32(0.0f);
   std::vector<uint32_t> idx(count);
   for (uint32_t i = 0; i < count; i++)
      idx[i] = P.constant(U32, i);

   if (!SV) {
      uint32_t AR = P.array(F32, count);
      uint32_t PAR = P.ptr(StInput, AR);
      SV = m.new_id();
      P.list.push_back(Inst(OpVariable_, {PAR, SV, StInput}));
      added_var = true;
   } else {
      for (Inst &in : m.v) {
         if (in.op == OpDecorate_ && in.w.size() >= 3 && in.w[0] == SV &&
             in.w[1] == DecBuiltIn && in.w[2] == BiClipDistance) {
            in.w = {SV, DecLocation, loc};
            break;
         }
      }
   }

   /* the code that runs first in the entry function */
   std::vector<Inst> code;
   uint32_t any = 0;
   for (uint32_t i = 0; i < count; i++) {
      uint32_t p = m.new_id(), d = m.new_id(), l = m.new_id();
      code.push_back(Inst(OpAccessChain_, {PIF, p, SV, idx[i]}));
      code.push_back(Inst(OpLoad_, {F32, d, p}));
      code.push_back(Inst(OpFOrdLessThan_, {BOOL, l, d, ZERO}));
      if (!any) {
         any = l;
      } else {
         uint32_t o = m.new_id();
         code.push_back(Inst(OpLogicalOr_, {BOOL, o, any, l}));
         any = o;
      }
   }
   const uint32_t killL = m.new_id(), contL = m.new_id();
   code.push_back(Inst(OpSelectionMerge_, {contL, 0}));
   code.push_back(Inst(OpBranchConditional_, {any, killL, contL}));
   code.push_back(Inst(OpLabel_, {killL}));
   code.push_back(Inst(OpKill_, {}));
   code.push_back(Inst(OpLabel_, {contL}));

   /* find the first block of the entry function, skip its OpVariables */
   size_t fi = m.v.size();
   for (size_t k = 0; k < m.v.size(); k++)
      if (m.v[k].op == OpFunction_ && m.v[k].w.size() >= 2 && m.v[k].w[1] == e.func) {
         fi = k;
         break;
      }
   if (fi == m.v.size())
      return false;
   size_t at = fi + 1;
   while (at < m.v.size() && m.v[at].op == 55) /* OpFunctionParameter */
      at++;
   if (at >= m.v.size() || m.v[at].op != OpLabel_)
      return false;
   at++;
   while (at < m.v.size() && m.v[at].op == OpVariable_)
      at++;

   m.v.insert(m.v.begin() + at, code.begin(), code.end());
   m.index();
   P.commit();

   if (added_var) {
      m.v.insert(m.v.begin() + m.annotation_end(), Inst(OpDecorate_, {SV, DecLocation, loc}));
      m.index();
      if (!m.find_entry(WSPV_FRAGMENT, name, e))
         return false;
      m.v[e.inst].w.push_back(SV);
   }
   m.drop_capability_if_unused(CapClipDistance, BiClipDistance);
   return true;
}

} /* namespace */

/* ------------------------------------------------------------------ C API */

extern "C" bool
wrapper_spv_analyze(const uint32_t *code, size_t size_bytes, uint32_t exec_model,
                    const char *entry, struct wrapper_spv_stage_info *info)
{
   if (!code || !entry || !info || (size_bytes & 3))
      return false;
   Module m;
   if (!m.parse(code, size_bytes / 4))
      return false;
   return analyze_module(m, exec_model, entry, info);
}

extern "C" bool
wrapper_spv_lower_clip_vs(const uint32_t *code, size_t size_bytes, const char *entry,
                          uint32_t location, uint32_t **out, size_t *out_size_bytes)
{
   if (!code || !entry || !out || (size_bytes & 3))
      return false;
   Module m;
   if (!m.parse(code, size_bytes / 4))
      return false;
   if (!lower_builtin_vs(m, entry, BiClipDistance, CapClipDistance, location))
      return false;
   return emit(m, out, out_size_bytes);
}

extern "C" bool
wrapper_spv_lower_cull_vs(const uint32_t *code, size_t size_bytes, const char *entry,
                          uint32_t location, uint32_t **out, size_t *out_size_bytes)
{
   if (!code || !entry || !out || (size_bytes & 3))
      return false;
   Module m;
   if (!m.parse(code, size_bytes / 4))
      return false;
   if (!lower_builtin_vs(m, entry, BiCullDistance, CapCullDistance, location))
      return false;
   return emit(m, out, out_size_bytes);
}

extern "C" bool
wrapper_spv_lower_clip_fs(const uint32_t *code, size_t size_bytes, const char *entry,
                          uint32_t count, uint32_t location,
                          uint32_t **out, size_t *out_size_bytes)
{
   if (!code || !entry || !out || (size_bytes & 3))
      return false;
   Module m;
   if (!m.parse(code, size_bytes / 4))
      return false;
   if (!lower_clip_fs_module(m, entry, count, location))
      return false;
   return emit(m, out, out_size_bytes);
}

extern "C" void
wrapper_spv_free(uint32_t *p)
{
   free(p);
}
