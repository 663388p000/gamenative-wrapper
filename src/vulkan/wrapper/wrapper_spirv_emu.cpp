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
#include <unordered_set>
#include <vector>

namespace {

enum : uint32_t {
   SpvMagic = 0x07230203,
   OpName_ = 5, OpMemberName_ = 6, OpEntryPoint_ = 15, OpCapability_ = 17,
   OpTypeBool_ = 20, OpTypeInt_ = 21, OpTypeFloat_ = 22, OpTypeVector_ = 23,
   OpTypeMatrix_ = 24, OpTypeArray_ = 28, OpTypeStruct_ = 30, OpTypePointer_ = 32,
   OpConstant_ = 43,
   OpFunction_ = 54, OpFunctionCall_ = 57,
   OpVariable_ = 59, OpLoad_ = 61, OpStore_ = 62, OpCopyMemory_ = 63,
   OpCopyMemorySized_ = 64, OpAccessChain_ = 65, OpInBoundsAccessChain_ = 66,
   OpPtrAccessChain_ = 67, OpInBoundsPtrAccessChain_ = 70,
   OpCopyObject_ = 83, OpSelect_ = 169, OpPhi_ = 245,
   OpReturnValue_ = 254,
   OpDecorate_ = 71, OpMemberDecorate_ = 72,
   OpGroupDecorate_ = 74, OpGroupMemberDecorate_ = 75,
   OpLogicalOr_ = 166, OpFOrdLessThan_ = 184,
   OpSelectionMerge_ = 247, OpLabel_ = 248, OpBranchConditional_ = 250, OpKill_ = 252,
};
enum : uint32_t { DecBuiltIn = 11, DecLocation = 30 };
enum : uint32_t { BiClipDistance = 3, BiCullDistance = 4, BiLayer = 9, BiViewportIndex = 10 };
enum : uint32_t { StInput = 1, StOutput = 3 };
enum : uint32_t {
   CapFloat16 = 9,
   CapFloat64 = 10,
   CapInt64 = 11,
   CapInt16 = 22,
   CapClipDistance = 32,
   CapCullDistance = 33,
   CapStorageInputOutput16 = 4436,
};

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
   bool id_exhausted = false;
   std::vector<Inst> v;
   std::unordered_map<uint32_t, size_t> defs;

   bool parse(const uint32_t *code, size_t words)
   {
      id_exhausted = false;
      if (!code || words < 5 || code[0] != SpvMagic)
         return false;

      memcpy(hdr, code, sizeof(hdr));

      /* SPIR-V header: valid version, nonzero ID bound, zero schema. */
      if ((hdr[1] & 0xff0000ffu) != 0 ||
          hdr[1] > 0x00010600u ||
          hdr[3] == 0 || hdr[4] != 0)
         return false;

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
      {
         std::unordered_set<uint32_t> seen;
         for (const Inst &in : v) {
            uint32_t id;
            if (!result_id(in, id))
               continue;

            if (id == 0 || id >= hdr[3] || !seen.insert(id).second) {
               v.clear();
               defs.clear();
               return false;
            }
         }
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

   uint32_t new_id()
   {
      if (id_exhausted || hdr[3] == 0 || hdr[3] == UINT32_MAX) {
         id_exhausted = true;
         return 0;
      }
      return hdr[3]++;
   }

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
      const Inst *t;

      if (!d || d->op != OpConstant_ || d->w.size() != 3)
         return false;

      t = def(d->w[0]);
      if (!t || t->op != OpTypeInt_ || t->w.size() != 3 ||
          t->w[1] != 32 || (t->w[2] != 0 && t->w[2] != 1))
         return false;

      if (t->w[2] == 1 && (d->w[2] & 0x80000000u))
         return false;

      val = d->w[2];
      return true;
   }

   /* OpTypeArray -> element type + length */
   bool arr_info(uint32_t type, uint32_t &elem, uint32_t &n) const
   {
      const Inst *d = def(type);
      if (!d || d->op != OpTypeArray_ || d->w.size() != 3)
         return false;
      elem = d->w[1];
      return const_u32(d->w[2], n);
   }

   /* locations a type occupies in a stage interface; UNKNOWN if unsure */
   uint32_t slots(uint32_t type) const
   {
      const Inst *d = def(type);
      if (!d)
         return WSPV_SLOTS_UNKNOWN;

      switch (d->op) {
      case OpTypeBool_:
         return d->w.size() == 1 ? 1 : WSPV_SLOTS_UNKNOWN;

      case OpTypeInt_:
         return d->w.size() == 3 &&
                (d->w[1] == 32 || d->w[1] == 64) &&
                (d->w[2] == 0 || d->w[2] == 1)
                   ? 1 : WSPV_SLOTS_UNKNOWN;

      case OpTypeFloat_:
         return d->w.size() == 2 &&
                (d->w[1] == 32 || d->w[1] == 64)
                   ? 1 : WSPV_SLOTS_UNKNOWN;

      case OpTypeVector_: {
         if (d->w.size() != 3 || d->w[2] < 2 || d->w[2] > 4)
            return WSPV_SLOTS_UNKNOWN;

         const Inst *e = def(d->w[1]);
         if (!e)
            return WSPV_SLOTS_UNKNOWN;

         uint32_t width = 0;
         if (e->op == OpTypeInt_ && e->w.size() == 3 &&
             (e->w[2] == 0 || e->w[2] == 1))
            width = e->w[1];
         else if (e->op == OpTypeFloat_ && e->w.size() == 2)
            width = e->w[1];
         else
            return WSPV_SLOTS_UNKNOWN;

         if (width != 32 && width != 64)
            return WSPV_SLOTS_UNKNOWN;

         return width == 64 && d->w[2] >= 3 ? 2 : 1;
      }

      case OpTypeMatrix_: {
         if (d->w.size() != 3 || d->w[2] < 2 || d->w[2] > 4)
            return WSPV_SLOTS_UNKNOWN;

         uint32_t c = slots(d->w[1]);
         if (c == WSPV_SLOTS_UNKNOWN ||
             c > (WSPV_SLOTS_UNKNOWN - 1) / d->w[2])
            return WSPV_SLOTS_UNKNOWN;

         return c * d->w[2];
      }

      case OpTypeArray_: {
         uint32_t e, n;
         if (!arr_info(type, e, n) || n == 0)
            return WSPV_SLOTS_UNKNOWN;

         uint32_t s = slots(e);
         if (s == WSPV_SLOTS_UNKNOWN ||
             s > (WSPV_SLOTS_UNKNOWN - 1) / n)
            return WSPV_SLOTS_UNKNOWN;

         return s * n;
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
      if (!d || d->op != OpVariable_ || d->w.size() != 3)
         return false;
      const Inst *pt = def(d->w[0]);
      if (!pt || pt->op != OpTypePointer_ || pt->w.size() != 3 ||
          pt->w[1] != d->w[2])
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
      if (n == 0)
         return 0;

      auto length = [&](uint32_t len_id, uint32_t &val) {
         for (const std::vector<Inst> *src : {&m.v, &list}) {
            for (const Inst &in : *src) {
               if (in.op != OpConstant_ || in.w.size() != 3 ||
                   in.w[1] != len_id)
                  continue;

               const Inst *type = m.def(in.w[0]);

               if (!type) {
                  for (const Inst &candidate : list) {
                     if (candidate.op == OpTypeInt_ &&
                         candidate.w.size() == 3 &&
                         candidate.w[0] == in.w[0]) {
                        type = &candidate;
                        break;
                     }
                  }
               }

               if (!type || type->op != OpTypeInt_ ||
                   type->w.size() != 3 ||
                   type->w[1] != 32 ||
                   (type->w[2] != 0 && type->w[2] != 1))
                  return false;

               val = in.w[2];
               return true;
            }
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
      std::vector<Inst> declarations;
      std::vector<Inst> variables;

      /*
       * Preserve dependency order between types and constants.
       * Only global variables are moved to the end.
       */
      for (Inst &in : list) {
         if (in.op == OpVariable_)
            variables.push_back(std::move(in));
         else
            declarations.push_back(std::move(in));
      }

      /* New declarations must precede global variables that use them. */
      size_t decl_at = m.first_function();
      for (size_t i = 0; i < decl_at; i++) {
         if (m.v[i].op == OpVariable_) {
            decl_at = i;
            break;
         }
      }

      m.v.insert(m.v.begin() + decl_at,
                 declarations.begin(), declarations.end());

      /* Global variables belong immediately before the first function. */
      size_t var_at = m.first_function();
      m.v.insert(m.v.begin() + var_at,
                 variables.begin(), variables.end());

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
   case BiCullDistance: {
      const Inst *t;
      const bool is_clip = builtin == BiClipDistance;

      if (!m.arr_info(type, elem, n) || n == 0 || n > 8) {
         if (is_clip)
            out->clip_unknown = true;
         else
            out->cull_unknown = true;
         break;
      }

      t = m.def(elem);
      if (!t || t->op != OpTypeFloat_ || t->w.size() != 2 || t->w[1] != 32) {
         if (is_clip)
            out->clip_unknown = true;
         else
            out->cull_unknown = true;
         break;
      }

      if (is_clip) {
         out->has_clip = true;
         out->clip_count = n;
      } else {
         out->has_cull = true;
         out->cull_count = n;
      }
      break;
   }
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
               if (s == WSPV_SLOTS_UNKNOWN ||
                   s >= UINT32_MAX - loc)
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
      if (s == WSPV_SLOTS_UNKNOWN ||
          s >= UINT32_MAX - loc)
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
   struct Transaction {
      Module &target;
      Module backup;
      bool committed = false;

      explicit Transaction(Module &m) : target(m), backup(m) {}
      ~Transaction()
      {
         if (!committed)
            target = std::move(backup);
      }
   } transaction(m);

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

   /*
    * Preflight: the block transformation must not partially modify a
    * module when the original struct has shared users or group decorations.
    */
   if (k >= 0) {
      for (const Inst &in : m.v) {
      if (in.op == OpVariable_ && in.w.size() >= 3 && in.w[1] != V) {
         const Inst *pt = m.def(in.w[0]);
         if (pt && pt->op == OpTypePointer_ && pt->w.size() >= 3 &&
             pt->w[2] == S)
            return false;
      }

      if (in.op == OpGroupDecorate_ && in.w.size() >= 2) {
         for (size_t j = 1; j < in.w.size(); j++)
            if (in.w[j] == S)
               return false;
      }

      if (in.op == OpGroupMemberDecorate_ && in.w.size() >= 3) {
         for (size_t j = 1; j + 1 < in.w.size(); j += 2)
            if (in.w[j] == S)
               return false;
      }
      }
   }

   if (k == -2) {
      for (Inst &in : m.v) {
         if (in.op == OpDecorate_ && in.w.size() >= 3 && in.w[0] == V &&
             in.w[1] == DecBuiltIn && in.w[2] == builtin) {
            in.w = {V, DecLocation, loc};
            m.drop_capability_if_unused(cap, builtin);
            m.index();
            transaction.committed = true;
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
      if ((in.op == OpAccessChain_ || in.op == OpInBoundsAccessChain_) && in.w.size() >= 4) {
         chains[in.w[1]] = Chain{in.w[2], std::vector<uint32_t>(in.w.begin() + 3, in.w.end())};
      }
   }
   bool resolve_failed = false;
   auto resolve = [&](uint32_t p, std::vector<uint32_t> &idx) {
      std::vector<const Chain *> stack;
      size_t guard = 0;
      for (; guard <= chains.size(); guard++) {
         auto it = chains.find(p);
         if (it == chains.end())
            break;
         stack.push_back(&it->second);
         p = it->second.base;
      }

      idx.clear();
      for (auto it = stack.rbegin(); it != stack.rend(); ++it)
         idx.insert(idx.end(), (*it)->idx.begin(), (*it)->idx.end());

      /* A cycle or unresolved chain makes this lowering unsafe. */
      if (chains.find(p) != chains.end()) {
         resolve_failed = true;
         return 0u;
      }
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

      /* PtrAccessChain pointer arithmetic is not lowered by this pass. */
      if ((in.op == OpPtrAccessChain_ || in.op == OpInBoundsPtrAccessChain_) &&
          in.w.size() >= 4 && resolve(in.w[2], idx) == V)
         return false;

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
      } else if (in.op == OpCopyObject_ && in.w.size() >= 3) {
         if (resolve(in.w[2], idx) == V)
            return false;
      } else if (in.op == OpSelect_ && in.w.size() >= 5) {
         if (resolve(in.w[3], idx) == V || resolve(in.w[4], idx) == V)
            return false;
      } else if (in.op == OpPhi_ && in.w.size() >= 4) {
         for (size_t a = 2; a + 1 < in.w.size(); a += 2)
            if (resolve(in.w[a], idx) == V)
               return false;
      } else if (in.op == OpReturnValue_ && !in.w.empty()) {
         if (resolve(in.w[0], idx) == V)
            return false;
      } else if (in.op == OpFunctionCall_) {
         for (size_t a = 3; a < in.w.size(); a++)
            if (resolve(in.w[a], idx) == V)
               return false;
      }
   }

   if (resolve_failed)
      return false;

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

   /*
    * Clone the interface struct for V. The original struct may be shared
    * by other variables, so never remove a member from S itself.
    */
   m.index();

   VarInfo vi;
   if (!m.var_info(V, vi) || vi.pointee != S)
      return false;

   const Inst *orig_struct = m.def(S);
   if (!orig_struct || orig_struct->op != OpTypeStruct_ ||
       orig_struct->w.size() < (size_t)k + 2)
      return false;

   const uint32_t NS = m.new_id();
   const uint32_t PV = m.new_id();

   Inst cloned_struct = *orig_struct;
   cloned_struct.w[0] = NS;
   cloned_struct.w.erase(cloned_struct.w.begin() + 1 + k);

   /* Keep all generated type declarations before generated constants. */
   size_t type_at = 0;
   while (type_at < P.list.size() && P.list[type_at].op != OpConstant_)
      type_at++;

   P.list.insert(P.list.begin() + type_at, std::move(cloned_struct));
   P.list.insert(P.list.begin() + type_at + 1,
                 Inst(OpTypePointer_, {PV, vi.storage, NS}));

   /* Redirect only V to the cloned struct pointer type. */
   bool found_var = false;
   for (Inst &in : m.v) {
      if (in.op == OpVariable_ && in.w.size() >= 3 && in.w[1] == V) {
         in.w[0] = PV;
         found_var = true;
         break;
      }
   }
   if (!found_var)
      return false;

   /*
    * Copy names and decorations to the clone. Member indices after k
    * are shifted; decorations for the removed member are omitted.
    */
   std::vector<Inst> annotations;
   for (const Inst &in : m.v) {
      if (in.op == OpDecorate_ && in.w.size() >= 2 && in.w[0] == S) {
         Inst copy = in;
         copy.w[0] = NS;
         annotations.push_back(std::move(copy));
      } else if ((in.op == OpMemberDecorate_ || in.op == OpMemberName_) &&
                 in.w.size() >= 2 && in.w[0] == S) {
         if (in.w[1] == (uint32_t)k)
            continue;
         Inst copy = in;
         copy.w[0] = NS;
         if (copy.w[1] > (uint32_t)k)
            copy.w[1]--;
         annotations.push_back(std::move(copy));
      } else if (in.op == OpName_ && in.w.size() >= 1 && in.w[0] == S) {
         Inst copy = in;
         copy.w[0] = NS;
         annotations.push_back(std::move(copy));
      } else if (in.op == OpGroupDecorate_ && in.w.size() >= 2) {
         for (size_t j = 1; j < in.w.size(); j++)
            if (in.w[j] == S)
               return false;
      } else if (in.op == OpGroupMemberDecorate_ && in.w.size() >= 3) {
         for (size_t j = 1; j + 1 < in.w.size(); j += 2)
            if (in.w[j] == S)
               return false;
      }
   }

   annotations.push_back(Inst(OpDecorate_, {SV, DecLocation, loc}));

   size_t annotation_at = m.annotation_end();
   m.v.insert(m.v.begin() + annotation_at,
              annotations.begin(), annotations.end());

   /* Add the new output variable with the other global declarations. */
   P.list.push_back(Inst(OpVariable_, {PA, SV, StOutput}));

   if (m.id_exhausted)
      return false;
   P.commit();

   if (!m.find_entry(WSPV_VERTEX, name, e))
      return false;
   m.v[e.inst].w.push_back(SV);

   /*
    * The original struct must no longer describe a live interface variable.
    * Otherwise removing its BuiltIn decoration would affect another variable.
    */
   m.index();
   for (const Inst &in : m.v) {
      if (in.op != OpVariable_ || in.w.size() < 3 || in.w[1] == V)
         continue;

      const Inst *pt = m.def(in.w[0]);
      if (pt && pt->op == OpTypePointer_ && pt->w.size() >= 3 &&
          pt->w[2] == S)
         return false;
   }

   /* V now uses the cloned struct; remove the stale BuiltIn annotation. */
   for (size_t i = 0; i < m.v.size(); ) {
      const Inst &in = m.v[i];
      if (in.op == OpMemberDecorate_ && in.w.size() >= 4 &&
          in.w[0] == S && in.w[1] == (uint32_t)k &&
          in.w[2] == DecBuiltIn && in.w[3] == builtin) {
         m.v.erase(m.v.begin() + i);
         continue;
      }
      i++;
   }

   m.index();
   m.drop_capability_if_unused(cap, builtin);
   transaction.committed = true;
   return true;
}

/* ------------------------------------------------------------ FS lowering */

static bool
lower_clip_fs_module(Module &m, const char *name, uint32_t count, uint32_t loc)
{
   struct Transaction {
      Module &target;
      Module backup;
      bool committed = false;

      explicit Transaction(Module &m) : target(m), backup(m) {}
      ~Transaction()
      {
         if (!committed)
            target = std::move(backup);
      }
   } transaction(m);

   Entry e;
   if (!m.find_entry(WSPV_FRAGMENT, name, e) || count == 0 || count > 8)
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

               const Inst *elem_type = m.def(el);
               if (!elem_type || elem_type->op != OpTypeFloat_ ||
                   elem_type->w.size() != 2 || elem_type->w[1] != 32)
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
   if (m.id_exhausted)
      return false;
   P.commit();

   if (added_var) {
      m.v.insert(m.v.begin() + m.annotation_end(), Inst(OpDecorate_, {SV, DecLocation, loc}));
      m.index();
      if (!m.find_entry(WSPV_FRAGMENT, name, e))
         return false;
      m.v[e.inst].w.push_back(SV);
   }
   m.drop_capability_if_unused(CapClipDistance, BiClipDistance);
   transaction.committed = true;
   return true;
}


/* ------------------------------------------------------- geometry shader */

enum : uint32_t {
   OpExtension_ = 10, OpMemoryModel_ = 14, OpExecutionMode_ = 16,
   OpTypeVoid_ = 19, OpTypeFunction_ = 33,
   OpFunctionEnd_ = 56,
   OpLogicalAnd_ = 167,
   OpEmitVertex_ = 218, OpEndPrimitive_ = 219,
   OpBranch_ = 249, OpReturn_ = 253,
};
enum : uint32_t {
   DecBlock = 2, DecNoPerspective = 13, DecFlat = 14, DecCentroid = 16, DecComponent = 31,
   DecRelaxedPrecision = 0,
};
enum : uint32_t { BiPosition = 0, BiPointSize = 1, BiPrimitiveId = 7 };
enum : uint32_t { StPrivate = 6 };
enum : uint32_t {
   CapShader = 1, CapGeometry = 2, CapMultiViewport = 57, CapShaderLayer = 69,
   CapShaderViewportIndex = 70, CapGeometryPointSize = 24,
   CapShaderViewportIndexLayerEXT = 5254,
};
enum : uint32_t {
   ExecInvocations = 0, ExecOutputVertices = 26, ExecOutputPoints = 27,
   ExecOutputLineStrip = 28, ExecOutputTriangleStrip = 29,
   ExecInputPoints = 19, ExecInputLines = 21, ExecInputTriangles = 22,
};
enum : uint32_t { ModelGeometry = 3 };

static std::vector<uint32_t>
str_words(const char *s)
{
   std::vector<uint32_t> w;
   size_t n = strlen(s) + 1;
   w.assign((n + 3) / 4, 0);
   memcpy(w.data(), s, n);
   return w;
}

/* Interface scalar types supported by the generated pass-through GS. */
static bool
gs_scalar_ok(const Module &m, uint32_t type,
             const wrapper_spv_gs_params &p)
{
   const Inst *d = m.def(type);
   if (!d)
      return false;

   if (d->op == OpTypeInt_) {
      if (d->w.size() != 3 ||
          (d->w[2] != 0 && d->w[2] != 1))
         return false;

      switch (d->w[1]) {
      case 16:
         return p.shader_int16 && p.storage_input_output16;
      case 32:
         return true;
      case 64:
         return p.shader_int64;
      default:
         return false;
      }
   }

   if (d->op == OpTypeFloat_) {
      if (d->w.size() != 2)
         return false;

      switch (d->w[1]) {
      case 16:
         return p.shader_float16 && p.storage_input_output16;
      case 32:
         return true;
      case 64:
         return p.shader_float64;
      default:
         return false;
      }
   }

   return false;
}

static bool
gs_type_ok(const Module &m, uint32_t type,
           const wrapper_spv_gs_params &p, int depth = 0)
{
   const Inst *d = m.def(type);
   uint32_t len;

   if (!d || depth > 4)
      return false;

   switch (d->op) {
   case OpTypeInt_:
   case OpTypeFloat_:
      return gs_scalar_ok(m, type, p);

   case OpTypeVector_:
      return d->w.size() == 3 &&
             d->w[2] >= 2 && d->w[2] <= 4 &&
             gs_scalar_ok(m, d->w[1], p);

   case OpTypeMatrix_: {
      if (d->w.size() != 3 ||
          d->w[2] < 2 || d->w[2] > 4)
         return false;

      const Inst *col = m.def(d->w[1]);
      if (!col || col->op != OpTypeVector_ ||
          col->w.size() != 3 ||
          col->w[2] < 2 || col->w[2] > 4)
         return false;

      const Inst *scalar = m.def(col->w[1]);
      return scalar && scalar->op == OpTypeFloat_ &&
             gs_scalar_ok(m, col->w[1], p);
   }

   case OpTypeArray_:
      return d->w.size() == 3 &&
             m.const_u32(d->w[2], len) && len > 0 &&
             gs_type_ok(m, d->w[1], p, depth + 1);

   default:
      return false;
   }
}

struct GsUser {
   uint32_t var, type, location;
   std::vector<Inst> out_decos;    /* decorations the output variable keeps */
   std::vector<uint32_t> in_decos; /* Component, if any (Location is added) */
   uint32_t in_var = 0;
};

struct GsGen {
   const Module &vs;
   uint32_t bound;
   bool id_exhausted = false;
   std::vector<Inst> globals;      /* types, constants, variables, definition order */
   std::unordered_map<uint32_t, bool> copied;
   std::unordered_map<uint32_t, bool> copying;

   explicit GsGen(const Module &m) : vs(m), bound(m.hdr[3]) {}

   uint32_t new_id()
   {
      if (id_exhausted || bound == 0 || bound == UINT32_MAX) {
         id_exhausted = true;
         return 0;
      }
      return bound++;
   }

   /* a type of the vertex shader (and everything it needs), once */
   bool copy_type(uint32_t id, unsigned depth = 0)
   {
      if (depth > 4)
         return false;
      if (copied.count(id))
         return true;
      const Inst *d = vs.def(id);
      if (!d || copying.count(id))
         return false;

      copying[id] = true;
      struct CopyGuard {
         std::unordered_map<uint32_t, bool> &active;
         uint32_t id;
         ~CopyGuard() { active.erase(id); }
      } guard{copying, id};

      switch (d->op) {
      case OpTypeInt_:
         if (d->w.size() != 3 ||
             (d->w[1] != 16 && d->w[1] != 32 && d->w[1] != 64) ||
             (d->w[2] != 0 && d->w[2] != 1))
            return false;
         break;
      case OpTypeFloat_:
         if (d->w.size() != 2 ||
             (d->w[1] != 16 && d->w[1] != 32 && d->w[1] != 64))
            return false;
         break;
      case OpTypeVector_: {
         if (d->w.size() != 3 || d->w[2] < 2 || d->w[2] > 4)
            return false;

         const Inst *component = vs.def(d->w[1]);
         if (!component ||
             !((component->op == OpTypeInt_ && component->w.size() == 3 &&
                (component->w[1] == 16 || component->w[1] == 32 ||
                 component->w[1] == 64) &&
                (component->w[2] == 0 || component->w[2] == 1)) ||
               (component->op == OpTypeFloat_ && component->w.size() == 2 &&
                (component->w[1] == 16 || component->w[1] == 32 ||
                 component->w[1] == 64))) ||
             !copy_type(d->w[1], depth + 1))
            return false;
         break;
      }

      case OpTypeMatrix_: {
         if (d->w.size() != 3 || d->w[2] < 2 || d->w[2] > 4)
            return false;

         const Inst *col = vs.def(d->w[1]);
         if (!col || col->op != OpTypeVector_ || col->w.size() != 3 ||
             col->w[2] < 2 || col->w[2] > 4)
            return false;

         const Inst *scalar = vs.def(col->w[1]);
         if (!scalar || scalar->op != OpTypeFloat_ ||
             scalar->w.size() != 2 ||
             (scalar->w[1] != 16 && scalar->w[1] != 32 &&
              scalar->w[1] != 64) ||
             !copy_type(d->w[1], depth + 1))
            return false;
         break;
      }

      case OpTypeArray_: {
         const Inst *c;
         uint32_t len;
         if (d->w.size() != 3 ||
             !vs.const_u32(d->w[2], len) || len == 0 ||
             !copy_type(d->w[1], depth + 1))
            return false;
         c = vs.def(d->w[2]);
         if (!c || c->op != OpConstant_ || c->w.size() != 3 ||
             !copy_type(c->w[0], depth + 1))
            return false;
         if (!copied.count(d->w[2])) {
            globals.push_back(*c);
            copied[d->w[2]] = true;
         }
         break;
      }
      default:
         return false;
      }
      globals.push_back(*d);
      copied[id] = true;
      return true;
   }

   /* find a type/constant already in `globals` (non-aggregate types must not be duplicated) */
   uint32_t find(uint32_t op, std::initializer_list<uint32_t> tail, size_t id_pos = 0) const
   {
      for (const Inst &in : globals) {
         if (in.op != op || in.w.size() != tail.size() + 1)
            continue;
         size_t j = 0;
         bool eq = true;
         for (uint32_t t : tail) {
            size_t pos = j < id_pos ? j : j + 1;
            if (in.w[pos] != t) {
               eq = false;
               break;
            }
            j++;
         }
         if (eq)
            return in.w[id_pos];
      }
      return 0;
   }

   uint32_t type_simple(uint32_t op, std::initializer_list<uint32_t> tail)
   {
      uint32_t id = find(op, tail);
      if (!id) {
         id = new_id();
         Inst in;
         in.op = op;
         in.w.push_back(id);
         for (uint32_t t : tail)
            in.w.push_back(t);
         globals.push_back(in);
      }
      return id;
   }

   uint32_t f32() { return type_simple(OpTypeFloat_, {32}); }
   uint32_t u32() { return type_simple(OpTypeInt_, {32, 0}); }
   uint32_t boolean() { return type_simple(OpTypeBool_, {}); }
   uint32_t vec4() { return type_simple(OpTypeVector_, {f32(), 4}); }
   uint32_t ptr(uint32_t storage, uint32_t pointee)
   {
      return type_simple(OpTypePointer_, {storage, pointee});
   }

   uint32_t constant(uint32_t type, uint32_t bits)
   {
      for (const Inst &in : globals)
         if (in.op == OpConstant_ && in.w.size() == 3 && in.w[0] == type && in.w[2] == bits)
            return in.w[1];
      uint32_t id = new_id();
      globals.push_back(Inst(OpConstant_, {type, id, bits}));
      return id;
   }

   uint32_t constant_u32(uint32_t v) { return constant(u32(), v); }

   uint32_t array_of(uint32_t elem, uint32_t n)
   {
      uint32_t len = constant_u32(n);
      for (const Inst &in : globals)
         if (in.op == OpTypeArray_ && in.w.size() == 3 && in.w[1] == elem && in.w[2] == len)
            return in.w[0];
      uint32_t id = new_id();
      globals.push_back(Inst(OpTypeArray_, {id, elem, len}));
      return id;
   }
};

struct GsIface {
   bool has_pos = false, has_psize = false;
   std::vector<GsUser> users;
};

static bool
gs_builtin_type_ok(const Module &m, uint32_t builtin, uint32_t type)
{
   const Inst *t = m.def(type);

   if (builtin == BiPosition) {
      if (!t || t->op != OpTypeVector_ || t->w.size() != 3 ||
          t->w[2] != 4)
         return false;

      t = m.def(t->w[1]);
      return t && t->op == OpTypeFloat_ &&
             t->w.size() == 2 && t->w[1] == 32;
   }

   if (builtin == BiPointSize)
      return t && t->op == OpTypeFloat_ &&
             t->w.size() == 2 && t->w[1] == 32;

   return false;
}

static bool
gs_collect(const Module &m, const Entry &e, GsIface &g, const wrapper_spv_gs_params &p)
{
   const Inst &ep = m.v[e.inst];

   for (size_t k = e.iface; k < ep.w.size(); k++) {
      VarInfo vi;
      uint32_t bi, loc;

      if (!m.var_info(ep.w[k], vi) || vi.storage != StOutput)
         continue;

      if (m.decor(vi.id, DecBuiltIn, &bi)) {
         if (!gs_builtin_type_ok(m, bi, vi.pointee))
            return false;
         if (bi == BiPosition)
            g.has_pos = true;
         else if (bi == BiPointSize)
            g.has_psize = true;
         else
            return false;    /* ClipDistance, CullDistance, Layer, ViewportIndex, ... */
         continue;
      }

      const Inst *t = m.def(vi.pointee);
      if (t && t->op == OpTypeStruct_) {
         /* Validate the complete structure before inspecting its members. */
         if (t->w.size() < 2)
            return false;
         for (size_t mi = 1; mi < t->w.size(); mi++)
            if (!m.def(t->w[mi]))
               return false;

         /* gl_PerVertex style block: Position / PointSize members only */
         for (uint32_t mi = 0; mi + 1 < t->w.size(); mi++) {
            if (!m.member_decor(t->w[0], mi, DecBuiltIn, &bi) ||
                !gs_builtin_type_ok(m, bi, t->w[1 + mi]))
               return false;
            if (bi == BiPosition)
               g.has_pos = true;
            else if (bi == BiPointSize)
               g.has_psize = true;
            else
               return false;
         }
         continue;
      }

      if (!m.decor(vi.id, DecLocation, &loc) || !gs_type_ok(m, vi.pointee, p))
         return false;

      GsUser u;
      u.var = vi.id;
      u.type = vi.pointee;
      u.location = loc;
      for (const Inst &in : m.v) {
         if (in.op != OpDecorate_ || in.w.size() < 2 || in.w[0] != vi.id)
            continue;
         switch (in.w[1]) {
         case DecLocation:
            if (in.w.size() != 3)
               return false;
            u.out_decos.push_back(in);
            break;
         case DecFlat:
         case DecNoPerspective:
         case DecCentroid:
         case DecRelaxedPrecision:
            if (in.w.size() != 2)
               return false;
            u.out_decos.push_back(in);
            break;
         case DecComponent:
            if (in.w.size() != 3 || in.w[2] > 3)
               return false;
            u.out_decos.push_back(in);
            u.in_decos.push_back(in.w[2]);
            break;
         case DecBuiltIn:
            return false;
         default:
            return false;       /* XFB, Index, ... : not something we can forward */
         }
      }
      g.users.push_back(std::move(u));
   }
   return g.has_pos;
}

static bool
make_gs(const Module &vs, const char *entry, const wrapper_spv_gs_params &p,
        std::vector<uint32_t> &out)
{
   Entry e;
   GsIface g;

   uint32_t memory_model_count = 0;
   for (const Inst &in : vs.v) {
      if (in.op != OpMemoryModel_)
         continue;

      if (in.w.size() != 2 || ++memory_model_count > 1 ||
          in.w[0] != 0 || in.w[1] != 1)
         return false;
   }
   if (memory_model_count != 1)
      return false;

   if (p.in_prim < 1 || p.in_prim > 3 || p.out_mode > WSPV_GS_POINTS)
      return false;
   if (p.out_mode != WSPV_GS_PASS && p.in_prim != 3)
      return false;
   if (!vs.find_entry(WSPV_VERTEX, entry, e) || !gs_collect(vs, e, g, p))
      return false;

   const uint32_t nverts = p.in_prim;                 /* vertices per input primitive */
   const uint32_t max_out = p.out_mode == WSPV_GS_WIRE ? 4 : nverts;

   /* the cull varying produced by wrapper_spv_lower_cull_vs() is consumed here
    * and not forwarded to the fragment shader */
   if (p.cull_location != WSPV_GS_NO_CULL &&
       (p.cull_count == 0 || p.cull_count > 8))
      return false;

   const bool cull = p.cull_location != WSPV_GS_NO_CULL;
   int cull_idx = -1;
   if (cull) {
      for (size_t i = 0; i < g.users.size(); i++) {
         uint32_t elem, n;
         const Inst *el;
         if (g.users[i].location == p.cull_location &&
             vs.arr_info(g.users[i].type, elem, n) && n == p.cull_count &&
             p.cull_count <= 8 &&
             (el = vs.def(elem)) && el->op == OpTypeFloat_ &&
             el->w.size() == 2 && el->w[1] == 32) {
            cull_idx = (int)i;
            break;
         }
      }
      if (cull_idx < 0)
         return false;
   }
   const bool psize = g.has_psize && p.forward_point_size;

   GsGen gen(vs);
   for (const GsUser &u : g.users)
      if (!gen.copy_type(u.type))
         return false;

   const uint32_t t_f32 = gen.f32(), t_u32 = gen.u32(), t_vec4 = gen.vec4();
   const uint32_t t_void = gen.type_simple(OpTypeVoid_, {});
   const uint32_t t_fn = gen.type_simple(OpTypeFunction_, {t_void});
   const uint32_t t_bool = cull ? gen.boolean() : 0;

   /* per-vertex block: gl_in[] / gl_out */
   const uint32_t blk = gen.new_id();
   {
      Inst s;
      s.op = OpTypeStruct_;
      s.w = {blk, t_vec4};
      if (psize)
         s.w.push_back(t_f32);
      gen.globals.push_back(s);
   }
   const uint32_t blk_arr = gen.array_of(blk, nverts);
   const uint32_t gl_in = gen.new_id(), gl_out = gen.new_id();
   gen.globals.push_back(Inst(OpVariable_, {gen.ptr(StInput, blk_arr), gl_in, StInput}));
   gen.globals.push_back(Inst(OpVariable_, {gen.ptr(StOutput, blk), gl_out, StOutput}));

   /* inputs: one [nverts] array per forwarded varying (and the cull distances) */
   for (GsUser &u : g.users) {
      u.in_var = gen.new_id();
      gen.globals.push_back(Inst(OpVariable_,
         {gen.ptr(StInput, gen.array_of(u.type, nverts)), u.in_var, StInput}));
   }

   /* outputs: the vertex shader's own variables, same type, same decorations */
   for (size_t i = 0; i < g.users.size(); i++) {
      if ((int)i == cull_idx)
         continue;
      gen.globals.push_back(Inst(OpVariable_,
         {gen.ptr(StOutput, g.users[i].type), g.users[i].var, StOutput}));
   }

   /* gl_PrimitiveID passes through (a fragment shader may read it) */
   const uint32_t pid_in = gen.new_id(), pid_out = gen.new_id();
   gen.globals.push_back(Inst(OpVariable_, {gen.ptr(StInput, t_u32), pid_in, StInput}));
   gen.globals.push_back(Inst(OpVariable_, {gen.ptr(StOutput, t_u32), pid_out, StOutput}));

   const uint32_t p_in_vec4 = gen.ptr(StInput, t_vec4), p_out_vec4 = gen.ptr(StOutput, t_vec4);
   const uint32_t p_in_f32 = gen.ptr(StInput, t_f32), p_out_f32 = gen.ptr(StOutput, t_f32);
   std::vector<uint32_t> p_in_user(g.users.size());
   for (size_t i = 0; i < g.users.size(); i++)
      p_in_user[i] = gen.ptr(StInput, g.users[i].type);

   std::vector<uint32_t> cv(std::max<uint32_t>(nverts, 2));
   for (uint32_t i = 0; i < cv.size(); i++)
      cv[i] = gen.constant_u32(i);
   std::vector<uint32_t> ck;
   for (uint32_t k = 0; cull && k < p.cull_count; k++)
      ck.push_back(gen.constant_u32(k));
   const uint32_t zero_f = cull ? gen.constant(t_f32, 0) : 0;

   /* ------------------------------------------------------------------ body */
   std::vector<Inst> body;
   auto I = [&](uint32_t op, std::initializer_list<uint32_t> w) { body.push_back(Inst(op, w)); };

   const uint32_t main_id = gen.new_id();
   I(OpLabel_, {gen.new_id()});

   /* a primitive is culled when, for one plane, every vertex is negative */
   uint32_t merge_label = 0;
   if (cull) {
      const GsUser &cu = g.users[cull_idx];
      uint32_t culled = 0;
      for (uint32_t k = 0; k < p.cull_count; k++) {
         uint32_t all_neg = 0;
         for (uint32_t i = 0; i < nverts; i++) {
            uint32_t ptr_id = gen.new_id(), val = gen.new_id(), neg = gen.new_id();
            I(OpAccessChain_, {p_in_f32, ptr_id, cu.in_var, cv[i], ck[k]});
            I(OpLoad_, {t_f32, val, ptr_id});
            I(OpFOrdLessThan_, {t_bool, neg, val, zero_f});
            if (!all_neg) {
               all_neg = neg;
            } else {
               uint32_t a = gen.new_id();
               I(OpLogicalAnd_, {t_bool, a, all_neg, neg});
               all_neg = a;
            }
         }
         if (!culled) {
            culled = all_neg;
         } else {
            uint32_t o = gen.new_id();
            I(OpLogicalOr_, {t_bool, o, culled, all_neg});
            culled = o;
         }
      }
      merge_label = gen.new_id();
      uint32_t emit_label = gen.new_id();
      I(OpSelectionMerge_, {merge_label, 0});
      I(OpBranchConditional_, {culled, merge_label, emit_label});
      I(OpLabel_, {emit_label});
   }

   auto emit_vertex = [&](uint32_t i) {
      uint32_t a = gen.new_id(), v = gen.new_id(), b = gen.new_id();
      I(OpAccessChain_, {p_in_vec4, a, gl_in, cv[i], cv[0]});
      I(OpLoad_, {t_vec4, v, a});
      I(OpAccessChain_, {p_out_vec4, b, gl_out, cv[0]});
      I(OpStore_, {b, v});
      if (psize) {
         a = gen.new_id(); v = gen.new_id(); b = gen.new_id();
         I(OpAccessChain_, {p_in_f32, a, gl_in, cv[i], cv[1]});
         I(OpLoad_, {t_f32, v, a});
         I(OpAccessChain_, {p_out_f32, b, gl_out, cv[1]});
         I(OpStore_, {b, v});
      }
      for (size_t n = 0; n < g.users.size(); n++) {
         if ((int)n == cull_idx)
            continue;
         a = gen.new_id(); v = gen.new_id();
         I(OpAccessChain_, {p_in_user[n], a, g.users[n].in_var, cv[i]});
         I(OpLoad_, {g.users[n].type, v, a});
         I(OpStore_, {g.users[n].var, v});
      }
      v = gen.new_id();
      I(OpLoad_, {t_u32, v, pid_in});
      I(OpStore_, {pid_out, v});
      I(OpEmitVertex_, {});
   };

   switch (p.out_mode) {
   case WSPV_GS_PASS:
      for (uint32_t i = 0; i < nverts; i++)
         emit_vertex(i);
      I(OpEndPrimitive_, {});
      break;
   case WSPV_GS_WIRE:                      /* v0 v1 v2 v0: the three edges */
      emit_vertex(0);
      emit_vertex(1);
      emit_vertex(2);
      emit_vertex(0);
      I(OpEndPrimitive_, {});
      break;
   default:                                /* WSPV_GS_POINTS: the three corners */
      for (uint32_t i = 0; i < 3; i++) {
         emit_vertex(i);
         I(OpEndPrimitive_, {});
      }
      break;
   }

   if (cull) {
      I(OpBranch_, {merge_label});
      I(OpLabel_, {merge_label});
   }
   I(OpReturn_, {});

   /* ID exhaustion makes the generated module invalid. */
   if (gen.id_exhausted || gen.bound == 0)
      return false;

   /* -------------------------------------------------------------- module */
   Module om;
   om.hdr[0] = SpvMagic;
   om.hdr[1] = vs.hdr[1];
   om.hdr[2] = 0;
   om.hdr[3] = gen.bound;
   om.hdr[4] = 0;
   std::vector<Inst> &mod = om.v;

   mod.push_back(Inst(OpCapability_, {CapShader}));
   mod.push_back(Inst(OpCapability_, {CapGeometry}));

   bool has_int16 = false;
   bool has_float16 = false;
   bool has_int64 = false;
   bool has_float64 = false;

   for (const Inst &in : gen.globals) {
      if (in.op == OpTypeInt_ && in.w.size() == 3) {
         has_int16 |= in.w[1] == 16;
         has_int64 |= in.w[1] == 64;
      } else if (in.op == OpTypeFloat_ && in.w.size() == 2) {
         has_float16 |= in.w[1] == 16;
         has_float64 |= in.w[1] == 64;
      }
   }

   if (has_int16)
      mod.push_back(Inst(OpCapability_, {CapInt16}));
   if (has_float16)
      mod.push_back(Inst(OpCapability_, {CapFloat16}));
   if (has_int64)
      mod.push_back(Inst(OpCapability_, {CapInt64}));
   if (has_float64)
      mod.push_back(Inst(OpCapability_, {CapFloat64}));
   if (has_int16 || has_float16)
      mod.push_back(Inst(OpCapability_, {CapStorageInputOutput16}));

   if (psize)
      mod.push_back(Inst(OpCapability_, {CapGeometryPointSize}));
   for (const Inst &in : vs.v)
      if (in.op == OpMemoryModel_)
         mod.push_back(in);

   {
      Inst ep;
      ep.op = OpEntryPoint_;
      ep.w = {ModelGeometry, main_id};
      std::vector<uint32_t> name = str_words("main");
      ep.w.insert(ep.w.end(), name.begin(), name.end());
      ep.w.push_back(gl_in);
      ep.w.push_back(gl_out);
      ep.w.push_back(pid_in);
      ep.w.push_back(pid_out);
      for (size_t i = 0; i < g.users.size(); i++) {
         ep.w.push_back(g.users[i].in_var);
         if ((int)i != cull_idx)
            ep.w.push_back(g.users[i].var);
      }
      mod.push_back(ep);
   }

   const uint32_t in_mode = p.in_prim == 1 ? ExecInputPoints :
                            p.in_prim == 2 ? ExecInputLines : ExecInputTriangles;
   uint32_t out_mode;
   if (p.out_mode == WSPV_GS_WIRE)
      out_mode = ExecOutputLineStrip;
   else if (p.out_mode == WSPV_GS_POINTS)
      out_mode = ExecOutputPoints;
   else
      out_mode = p.in_prim == 1 ? ExecOutputPoints :
                 p.in_prim == 2 ? ExecOutputLineStrip : ExecOutputTriangleStrip;
   mod.push_back(Inst(OpExecutionMode_, {main_id, in_mode}));
   mod.push_back(Inst(OpExecutionMode_, {main_id, ExecInvocations, 1}));
   mod.push_back(Inst(OpExecutionMode_, {main_id, out_mode}));
   mod.push_back(Inst(OpExecutionMode_, {main_id, ExecOutputVertices, max_out}));

   /* annotations */
   mod.push_back(Inst(OpDecorate_, {blk, DecBlock}));
   mod.push_back(Inst(OpMemberDecorate_, {blk, 0, DecBuiltIn, BiPosition}));
   if (psize)
      mod.push_back(Inst(OpMemberDecorate_, {blk, 1, DecBuiltIn, BiPointSize}));
   mod.push_back(Inst(OpDecorate_, {pid_in, DecBuiltIn, BiPrimitiveId}));
   mod.push_back(Inst(OpDecorate_, {pid_out, DecBuiltIn, BiPrimitiveId}));
   for (size_t i = 0; i < g.users.size(); i++) {
      const GsUser &u = g.users[i];
      mod.push_back(Inst(OpDecorate_, {u.in_var, DecLocation, u.location}));
      for (uint32_t c : u.in_decos)
         mod.push_back(Inst(OpDecorate_, {u.in_var, DecComponent, c}));
      if ((int)i != cull_idx)
         for (const Inst &d : u.out_decos)
            mod.push_back(d);
   }

   /*
    * Preserve type/constant dependency order, but keep all global
    * variables after the declarations they depend on.
    */
   {
      std::vector<Inst> declarations, variables;

      for (Inst &in : gen.globals) {
         if (in.op == OpVariable_)
            variables.push_back(std::move(in));
         else
            declarations.push_back(std::move(in));
      }

      mod.insert(mod.end(), declarations.begin(), declarations.end());
      mod.insert(mod.end(), variables.begin(), variables.end());
   }

   mod.push_back(Inst(OpFunction_, {t_void, main_id, 0, t_fn}));
   mod.insert(mod.end(), body.begin(), body.end());
   mod.push_back(Inst(OpFunctionEnd_, {}));

   om.hdr[3] = gen.bound;
   om.serialize(out);
   return true;
}

/* ---------------------------------------------- capability translations */

static std::string
ext_name(const Inst &in)
{
   std::string s;
   for (size_t j = 0; j < in.w.size(); j++) {
      for (int b = 0; b < 4; b++) {
         char c = (char)((in.w[j] >> (8 * b)) & 0xff);
         if (!c)
            return s;
         s.push_back(c);
      }
   }
   return s;
}

static std::string
ext_name_at(const Inst &in, size_t first_word)
{
   std::string s;
   for (size_t j = first_word; j < in.w.size(); j++) {
      for (int b = 0; b < 4; b++) {
         char c = (char)((in.w[j] >> (8 * b)) & 0xff);
         if (!c)
            return s;
         s.push_back(c);
      }
   }
   return s;
}

/* SPV_EXT_shader_viewport_index_layer -> the Vulkan 1.2 core capabilities */
static bool
translate_viewport_layer(Module &m)
{
   const bool uses_layer = m.has_builtin_decoration(BiLayer);
   const bool uses_vp = m.has_builtin_decoration(BiViewportIndex);
   std::vector<Inst> nv;
   bool found = false;

   nv.reserve(m.v.size() + 1);
   for (Inst &in : m.v) {
      if (in.op == OpCapability_ && !in.w.empty() && in.w[0] == CapShaderViewportIndexLayerEXT) {
         found = true;
         if (uses_vp)
            nv.push_back(Inst(OpCapability_, {CapShaderViewportIndex}));
         if (uses_layer)
            nv.push_back(Inst(OpCapability_, {CapShaderLayer}));
         continue;
      }
      if (in.op == OpExtension_ && ext_name(in) == "SPV_EXT_shader_viewport_index_layer")
         continue;
      nv.push_back(std::move(in));
   }
   if (!found)
      return false;
   m.v = std::move(nv);
   if (m.hdr[1] < 0x00010500)
      m.hdr[1] = 0x00010500;      /* ShaderLayer / ShaderViewportIndex are SPIR-V 1.5 */
   m.index();
   return true;
}

/* A geometry shader that writes gl_ViewportIndex needs the MultiViewport
 * capability, which only exists with multiViewport. Without the feature the
 * viewport index is dropped (everything lands in viewport 0): the variable
 * becomes a Private one, so the stores stay valid and do nothing. */
static bool
strip_viewport_index_gs(Module &m, const char *name)
{
   struct Transaction {
      Module &target;
      Module backup;
      bool committed = false;

      explicit Transaction(Module &m) : target(m), backup(m) {}
      ~Transaction()
      {
         if (!committed)
            target = std::move(backup);
      }
   } transaction(m);

   std::vector<uint32_t> vars;
   std::vector<size_t> entries;                 /* OpEntryPoint instructions involved */

   for (const Inst &in : m.v)
      if (in.op == OpMemberDecorate_ && in.w.size() >= 4 && in.w[2] == DecBuiltIn &&
          in.w[3] == BiViewportIndex)
         return false;                 /* member of a block: not handled */

   for (size_t k = 0; k < m.v.size(); k++) {
      const Inst &ep = m.v[k];
      Entry e;
      if (ep.op != OpEntryPoint_ || ep.w.size() < 3 || ep.w[0] != WSPV_GEOMETRY)
         continue;
      std::string nm = ext_name_at(ep, 2);
      if (name && nm != name)
         continue;
      if (!m.find_entry(WSPV_GEOMETRY, nm.c_str(), e))
         continue;
      entries.push_back(k);
      for (size_t j = e.iface; j < ep.w.size(); j++) {
         VarInfo vi;
         uint32_t bi;
         if (m.var_info(ep.w[j], vi) && vi.storage == StOutput &&
             m.decor(vi.id, DecBuiltIn, &bi) && bi == BiViewportIndex &&
             std::find(vars.begin(), vars.end(), vi.id) == vars.end())
            vars.push_back(vi.id);
      }
   }
   if (vars.empty())
      return false;

   std::unordered_map<uint32_t, uint32_t> ptr_for;   /* pointee -> Private pointer type */
   for (uint32_t var : vars) {
      VarInfo vi;
      auto it = m.defs.find(var);
      if (it == m.defs.end() || !m.var_info(var, vi))
         return false;
      uint32_t pp;
      auto f = ptr_for.find(vi.pointee);
      if (f != ptr_for.end()) {
         pp = f->second;
      } else {
         pp = 0;
         for (const Inst &in : m.v)
            if (in.op == OpTypePointer_ && in.w.size() == 3 && in.w[1] == StPrivate &&
                in.w[2] == vi.pointee)
               pp = in.w[0];
         if (!pp) {
            pp = m.new_id();
            if (!pp)
               return false;
            m.v.insert(m.v.begin() + it->second, Inst(OpTypePointer_, {pp, StPrivate, vi.pointee}));
            m.index();
         }
         ptr_for[vi.pointee] = pp;
      }
      Inst &vin = m.v[m.defs[var]];
      vin.w[0] = pp;
      vin.w[2] = StPrivate;
   }

   const bool listed_in_1_4 = m.hdr[1] >= 0x00010400;
   std::vector<Inst> nv;
   for (Inst &in : m.v) {
      if (in.op == OpDecorate_ && in.w.size() >= 3 && in.w[1] == DecBuiltIn &&
          in.w[2] == BiViewportIndex &&
          std::find(vars.begin(), vars.end(), in.w[0]) != vars.end())
         continue;
      if (in.op == OpEntryPoint_ && !listed_in_1_4) {
         Inst c = in;
         size_t j = 2;
         bool done = false;
         while (j < c.w.size() && !done) {
            for (int b = 0; b < 4; b++)
               if (!((c.w[j] >> (8 * b)) & 0xff)) { done = true; break; }
            j++;
         }
         for (size_t k = c.w.size(); k > j; k--)
            if (std::find(vars.begin(), vars.end(), c.w[k - 1]) != vars.end())
               c.w.erase(c.w.begin() + (k - 1));
         nv.push_back(std::move(c));
         continue;
      }
      nv.push_back(std::move(in));
   }
   m.v = std::move(nv);
   m.index();
   m.drop_capability_if_unused(CapMultiViewport, BiViewportIndex);
   if (!m.has_builtin_decoration(BiViewportIndex)) {
      for (size_t k = 0; k < m.v.size();) {
         if (m.v[k].op == OpCapability_ && !m.v[k].w.empty() &&
             m.v[k].w[0] == CapShaderViewportIndex) {
            m.v.erase(m.v.begin() + k);
         } else {
            k++;
         }
      }
      m.index();
   }
   if (m.id_exhausted)
      return false;

   transaction.committed = true;
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

extern "C" bool
wrapper_spv_make_gs(const uint32_t *code, size_t size_bytes, const char *entry,
                    const struct wrapper_spv_gs_params *params,
                    uint32_t **out, size_t *out_size_bytes)
{
   if (!code || !entry || !params || !out || (size_bytes & 3))
      return false;
   Module m;
   if (!m.parse(code, size_bytes / 4))
      return false;
   std::vector<uint32_t> words;
   if (!make_gs(m, entry, *params, words))
      return false;
   uint32_t *p = (uint32_t *)malloc(words.size() * sizeof(uint32_t));
   if (!p)
      return false;
   memcpy(p, words.data(), words.size() * sizeof(uint32_t));
   *out = p;
   *out_size_bytes = words.size() * sizeof(uint32_t);
   return true;
}

extern "C" uint32_t
wrapper_spv_scan(const uint32_t *code, size_t size_bytes)
{
   uint32_t flags = 0;
   size_t words = size_bytes / 4, i = 5;

   if (!code || words < 5 || code[0] != SpvMagic)
      return 0;
   /* capabilities and entry points all precede the first function */
   while (i < words) {
      uint32_t wc = code[i] >> 16, op = code[i] & 0xffff;
      if (wc == 0 || i + wc > words || op == OpFunction_)
         break;
      if (op == OpCapability_ && wc >= 2) {
         if (code[i + 1] == CapShaderViewportIndexLayerEXT)
            flags |= WSPV_SCAN_VIEWPORT_LAYER_EXT;
         else if (code[i + 1] == CapMultiViewport)
            flags |= WSPV_SCAN_MULTI_VIEWPORT;
         else if (code[i + 1] == CapClipDistance)
            flags |= WSPV_SCAN_CLIP_DISTANCE;
         else if (code[i + 1] == CapCullDistance)
            flags |= WSPV_SCAN_CULL_DISTANCE;
      } else if (op == OpDecorate_ && wc >= 4 &&
                 code[i + 2] == DecBuiltIn) {
         if (code[i + 3] == BiClipDistance)
            flags |= WSPV_SCAN_CLIP_BUILTIN;
         else if (code[i + 3] == BiCullDistance)
            flags |= WSPV_SCAN_CULL_BUILTIN;
      } else if (op == OpMemberDecorate_ && wc >= 5 &&
                 code[i + 3] == DecBuiltIn) {
         if (code[i + 4] == BiClipDistance)
            flags |= WSPV_SCAN_CLIP_BUILTIN;
         else if (code[i + 4] == BiCullDistance)
            flags |= WSPV_SCAN_CULL_BUILTIN;
      } else if (op == OpEntryPoint_ && wc >= 3) {
         switch (code[i + 1]) {
         case 0: flags |= WSPV_SCAN_VERTEX; break;
         case 1:
         case 2: flags |= WSPV_SCAN_TESS; break;
         case 3: flags |= WSPV_SCAN_GEOMETRY; break;
         case 4: flags |= WSPV_SCAN_FRAGMENT; break;
         case 5: flags |= WSPV_SCAN_COMPUTE; break;
         default: break;
         }
      }
      i += wc;
   }
   return flags;
}

extern "C" bool
wrapper_spv_translate_viewport_layer(const uint32_t *code, size_t size_bytes,
                                     uint32_t **out, size_t *out_size_bytes)
{
   if (!code || !out || (size_bytes & 3))
      return false;
   Module m;
   if (!m.parse(code, size_bytes / 4) || !translate_viewport_layer(m))
      return false;
   return emit(m, out, out_size_bytes);
}

extern "C" bool
wrapper_spv_strip_viewport_index_gs(const uint32_t *code, size_t size_bytes,
                                    const char *entry,
                                    uint32_t **out, size_t *out_size_bytes)
{
   if (!code || !out || (size_bytes & 3))
      return false;
   Module m;
   if (!m.parse(code, size_bytes / 4) || !strip_viewport_index_gs(m, entry))
      return false;
   return emit(m, out, out_size_bytes);
}
