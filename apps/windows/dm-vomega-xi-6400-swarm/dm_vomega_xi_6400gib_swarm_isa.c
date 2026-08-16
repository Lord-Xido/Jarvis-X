/*
 * DM-vOmegaXi+ 6400 GiB x 6400 GiB x 6400 GiB Virtual VRAM Swarm ISA Engine
 * Windows x86-64 PE32+, freestanding, no CRT.
 *
 * ISA word: [31:24] opcode | [23:20] rd/ra | [19:16] rs1/rv |
 *           [15:12] rs2 | [11:0] immediate/subop
 *
 * Arithmetic registers use deterministic signed Q16.16.  R10 (VOX_ADDR),
 * R13 (SYS_MODE), and R14 (SYNC/SCALAR) are typed 32-bit control values.
 * The 3D address fabric is virtual and sparse; no 6400^3 GiB allocation occurs.
 */

typedef void* HANDLE;
typedef unsigned long DWORD;
typedef int BOOL;
typedef unsigned long long U64;
typedef long long I64;
typedef unsigned int U32;
typedef int I32;

__declspec(dllimport) HANDLE __stdcall GetStdHandle(DWORD nStdHandle);
__declspec(dllimport) BOOL __stdcall WriteFile(HANDLE hFile, const void* lpBuffer, DWORD nNumberOfBytesToWrite, DWORD* lpNumberOfBytesWritten, void* lpOverlapped);
__declspec(dllimport) void __stdcall Sleep(DWORD dwMilliseconds);
__declspec(dllimport) void __stdcall ExitProcess(unsigned int uExitCode);

#define STD_OUTPUT_HANDLE ((DWORD)-11)
#define Q16_ONE 65536
#define Q16_HALF 32768
#define Q16_DT_001 655 /* nearest Q16.16 value to 0.01 = 0.0099945068 */
#define Q16_MIN ((I32)0x80000000u)
#define Q16_MAX ((I32)0x7fffffffu)

#define GIB_BYTES 1073741824ULL
#define AXIS_GIB 6400ULL
#define AXIS_BYTES (AXIS_GIB * GIB_BYTES)
#define PAGE_BYTES 65536ULL
#define AXIS_PAGES (AXIS_BYTES / PAGE_BYTES)
#define RESIDENT_PAGES 2048U
#define ACTIVE_AGENTS 64U
#define MAX_SCHED_CYCLES 6400U
#define CORE_Q16 (3200 * Q16_ONE)

#define OP_SYNC_SWARM    0x0FU
#define OP_VREAD3D       0x10U
#define OP_VWRITE3D      0x11U
#define OP_Q16MUL        0x20U
#define OP_Q16ADD        0x21U
#define OP_Q16SUB        0x22U
#define OP_EVAL_FGRAD    0x30U
#define OP_EVAL_FLATENT  0x31U
#define OP_EVAL_FREPEL   0x32U
#define OP_ENCODE_STEP   0x40U
#define OP_DECODE_STEP   0x41U
#define OP_HASH_ADDR     0x50U
#define OP_HALT          0xFFU

#define R_ZERO 0
#define R_POS_X 1
#define R_POS_Y 2
#define R_POS_Z 3
#define R_VEL_X 4
#define R_VEL_Y 5
#define R_VEL_Z 6
#define R_FRC_X 7
#define R_FRC_Y 8
#define R_FRC_Z 9
#define R_VOX_ADDR 10
#define R_LATENT_VAL 11
#define R_ERR_GRAD 12
#define R_SYS_MODE 13
#define R_SYNC_CTR 14
#define R_SCRATCH 15

static HANDLE g_out;

static U32 slen(const char* s){U32 n=0;while(s[n])++n;return n;}
static void out(const char* s){DWORD n=0;WriteFile(g_out,s,slen(s),&n,0);}
static void out_ch(char c){DWORD n=0;WriteFile(g_out,&c,1,&n,0);}
static void out_u64(U64 v){char b[32];U32 i=0;if(!v){out_ch('0');return;}while(v&&i<31){b[i++]=(char)('0'+(v%10ULL));v/=10ULL;}while(i)out_ch(b[--i]);}
static void out_i32(I32 v){I64 x=v;if(x<0){out_ch('-');x=-x;}out_u64((U64)x);}
static void out_hex32(U32 v){static const char h[]="0123456789ABCDEF";out("0x");for(int s=28;s>=0;s-=4)out_ch(h[(v>>s)&15]);}
static void out_q16(I32 v){I64 x=v;if(x<0){out_ch('-');x=-x;}U64 w=(U64)(x>>16),f=(U64)(x&0xFFFF),d=(f*10000ULL+32768ULL)>>16;if(d>=10000){w++;d-=10000;}out_u64(w);out_ch('.');if(d<1000)out_ch('0');if(d<100)out_ch('0');if(d<10)out_ch('0');out_u64(d);}
static void nl(void){out("\r\n");}
static void rule(void){out("--------------------------------------------------------------------------------\r\n");}

static I32 sat64(I64 x){if(x>2147483647LL)return 2147483647;if(x<-2147483648LL)return (I32)0x80000000u;return (I32)x;}
static I32 qadd(I32 a,I32 b){return sat64((I64)a+(I64)b);}
static I32 qsub(I32 a,I32 b){return sat64((I64)a-(I64)b);}
static I32 qmul(I32 a,I32 b){return sat64(((I64)a*(I64)b)>>16);}
static I32 qabs(I32 x){if(x==(I32)0x80000000u)return 0x7fffffff;return x<0?-x:x;}
static I32 qact(I32 x){I64 ax=qabs(x);I64 den=(I64)Q16_ONE+ax;return den?sat64(((I64)x*(I64)Q16_ONE)/den):0;}
static I32 qclamp(I32 x,I32 lo,I32 hi){return x<lo?lo:(x>hi?hi:x);}

static U32 op(U32 w){return w>>24;}
static U32 rd(U32 w){return (w>>20)&0xF;}
static U32 rs1(U32 w){return (w>>16)&0xF;}
static U32 rs2(U32 w){return (w>>12)&0xF;}
static U32 imm12(U32 w){return w&0xFFF;}
static U32 hash_rs3(U32 w){return (imm12(w)>>8)&0xF;}

static const U32 PROGRAM_CANONICAL_V1[] = {
    0x50A12300U, 0x10FA0000U, 0x30CF0000U, 0x31710000U,
    0x32810000U, 0x21778000U, 0x2177C000U, 0x21447000U,
    0x21114000U, 0x40BF0000U, 0x11AB0000U, 0x0F000000U,
    0xFF000000U
};
#define PROGRAM_CANONICAL_V1_WORDS 13U

static const U32 PROGRAM_ENCODE_DT001[] = {
    0x50A12300U,
    0x10FA0000U,
    0x30CF0000U,
    0x31710000U,
    0x32810000U,
    0x21778000U,
    0x2177C000U,
    0x2087E000U,
    0x21448000U,
    0x2084E000U,
    0x21118000U,
    0x40BF0000U,
    0x11AB0000U,
    0x0F000000U,
    0xFF000000U
};
#define PROGRAM_ENCODE_DT001_WORDS 15U

static const U32 PROGRAM_DECODE[] = {
    0x50A12300U,
    0x10FA0000U,
    0x41BF0000U,
    0x11AB0000U,
    0x0F000000U,
    0xFF000000U
};
#define PROGRAM_DECODE_WORDS 6U

typedef struct {
    U32 px,py,pz;
    U32 age;
    U32 valid;
    U32 committed;
    I32 value;
} Page;

static Page g_pages[RESIDENT_PAGES];
static U32 g_resident=0,g_evict_cursor=0,g_page_hits=0,g_page_faults=0,g_evictions=0;
static U64 g_logical_touched=0;

static U32 coord_to_page(I32 q){
    I64 lo=0,hi=(I64)AXIS_GIB*Q16_ONE,x=q;
    if(x<lo)x=lo;if(x>hi)x=hi;
    U64 p=((U64)x*(AXIS_PAGES-1ULL))/(U64)hi;
    return (U32)p;
}

static U32 find_or_touch(U32 px,U32 py,U32 pz,U32 cycle){
    for(U32 i=0;i<g_resident;i++){
        Page* p=&g_pages[i];
        if(p->valid&&p->px==px&&p->py==py&&p->pz==pz){p->age=cycle;g_page_hits++;return i;}
    }
    g_page_faults++;
    U32 slot;
    if(g_resident<RESIDENT_PAGES)slot=g_resident++;
    else{slot=g_evict_cursor++%RESIDENT_PAGES;g_evictions++;}
    g_pages[slot].px=px;g_pages[slot].py=py;g_pages[slot].pz=pz;g_pages[slot].age=cycle;
    g_pages[slot].valid=1;g_pages[slot].committed=0;g_pages[slot].value=0;
    g_logical_touched+=PAGE_BYTES;
    return slot;
}

typedef struct {
    I32 R[16];
    U32 pc;
    U32 waiting;
    U32 halted;
    U32 id;
    U32 epoch;
} Agent;

static Agent A[ACTIVE_AGENTS];
static U32 g_sched_cycles=0,g_epochs=0,g_vram_writes=0,g_verify_fail=0;

static void enforce_r0(Agent* a){a->R[R_ZERO]=0;}

static void init_agent(Agent* a,U32 id){
    for(U32 i=0;i<16;i++)a->R[i]=0;
    a->id=id;a->pc=0;a->waiting=0;a->halted=0;a->epoch=0;
    I32 dx=(I32)((int)(id%8)-4)*Q16_ONE*8;
    I32 dy=(I32)((int)((id/8)%8)-4)*Q16_ONE*8;
    I32 dz=(I32)((int)(id%5)-2)*Q16_ONE*6;
    a->R[R_POS_X]=CORE_Q16+dx;
    a->R[R_POS_Y]=CORE_Q16+dy;
    a->R[R_POS_Z]=CORE_Q16+dz;
    a->R[R_VEL_X]=(I32)((int)(id%3)-1)*1024;
    a->R[R_SYNC_CTR]=Q16_DT_001;
    a->R[R_SYS_MODE]=0;
    enforce_r0(a);
}

static I32 eval_fgrad(I32 voxel){return qmul(-8192,voxel);}
static I32 eval_flatent(I32 pos){return qmul(-1024,qsub(pos,CORE_Q16));}
static I32 eval_frepel(U32 id,I32 pos){
    I32 base=(id&1U)?768:-768;
    I32 side=pos>=CORE_Q16?1:-1;
    return side>0?base:-base;
}
static I32 encode_step(I32 voxel){return qact(voxel);}
static I32 decode_step(I32 latent){return qadd(latent,qmul(32768,latent));}

static const U32* agent_program(const Agent* a,U32* words){
    if(a->R[R_SYS_MODE]){*words=PROGRAM_DECODE_WORDS;return PROGRAM_DECODE;}
    *words=PROGRAM_ENCODE_DT001_WORDS;return PROGRAM_ENCODE_DT001;
}

static void exec_word(Agent* a,U32 w){
    U32 opcode=op(w),d=rd(w),s1=rs1(w),s2=rs2(w);
    switch(opcode){
        case OP_HASH_ADDR:{
            U32 s3=hash_rs3(w);
            U32 px=coord_to_page(a->R[s1]),py=coord_to_page(a->R[s2]),pz=coord_to_page(a->R[s3]);
            a->R[d]=(I32)find_or_touch(px,py,pz,g_sched_cycles);
            break;
        }
        case OP_VREAD3D:{
            U32 h=(U32)a->R[s1];
            a->R[d]=(h<g_resident&&g_pages[h].valid)?g_pages[h].value:0;
            break;
        }
        case OP_VWRITE3D:{
            U32 h=(U32)a->R[d];
            if(h<g_resident&&g_pages[h].valid){g_pages[h].value=a->R[s1];g_pages[h].committed=1;g_pages[h].age=g_sched_cycles;g_vram_writes++;}
            break;
        }
        case OP_Q16MUL:a->R[d]=qmul(a->R[s1],a->R[s2]);break;
        case OP_Q16ADD:a->R[d]=qadd(a->R[s1],a->R[s2]);break;
        case OP_Q16SUB:a->R[d]=qsub(a->R[s1],a->R[s2]);break;
        case OP_EVAL_FGRAD:a->R[d]=eval_fgrad(a->R[s1]);break;
        case OP_EVAL_FLATENT:a->R[d]=eval_flatent(a->R[s1]);break;
        case OP_EVAL_FREPEL:a->R[d]=eval_frepel(a->id,a->R[s1]);break;
        case OP_ENCODE_STEP:a->R[d]=encode_step(a->R[s1]);break;
        case OP_DECODE_STEP:a->R[d]=decode_step(a->R[s1]);break;
        case OP_SYNC_SWARM:a->waiting=1;break;
        case OP_HALT:a->halted=1;break;
        default:g_verify_fail++;a->halted=1;break;
    }
    enforce_r0(a);
}

static void step_agent(Agent* a){
    if(a->waiting||a->halted)return;
    U32 words=0;const U32* p=agent_program(a,&words);
    if(a->pc>=words){a->halted=1;return;}
    U32 w=p[a->pc];exec_word(a,w);
    if(!a->waiting&&!a->halted)a->pc++;
}

static U32 all_waiting_or_halted(void){
    for(U32 i=0;i<ACTIVE_AGENTS;i++)if(!A[i].waiting&&!A[i].halted)return 0;
    return 1;
}
static U32 all_halted(void){for(U32 i=0;i<ACTIVE_AGENTS;i++)if(!A[i].halted)return 0;return 1;}

static void release_barrier(void){
    if(!all_waiting_or_halted())return;
    for(U32 i=0;i<ACTIVE_AGENTS;i++)if(A[i].waiting){A[i].waiting=0;A[i].pc++;A[i].R[R_SYNC_CTR]=Q16_DT_001;}
}

static void next_epoch(void){
    g_epochs++;
    for(U32 i=0;i<ACTIVE_AGENTS;i++){
        Agent* a=&A[i];
        a->halted=0;a->waiting=0;a->pc=0;a->epoch++;
        a->R[R_SYS_MODE]=(I32)(a->epoch&1U);
        a->R[R_SYNC_CTR]=Q16_DT_001;
    }
}

static void print_program(void){
    out("Canonical V1 stream supplied by architecture:\r\n");
    for(U32 i=0;i<PROGRAM_CANONICAL_V1_WORDS;i++){
        out("  ");out_hex32(i*4);out("  ");out_hex32(PROGRAM_CANONICAL_V1[i]);nl();
    }
    out("\r\nExecutable V1.1 adds dt=0.01 Q16MUL scaling while preserving 32-bit format.\r\n");
}

static void telemetry(void){
    Agent* a=&A[0];
    out("cycle=");out_u64(g_sched_cycles);
    out(" epoch=");out_u64(g_epochs);
    out(" mode=");out(a->R[R_SYS_MODE]?"DECODE":"ENCODE");
    out(" pages=");out_u64(g_resident);
    out(" vram_writes=");out_u64(g_vram_writes);
    out(" faults=");out_u64(g_page_faults);
    out(" hits=");out_u64(g_page_hits);
    out(" evict=");out_u64(g_evictions);nl();
    out("A0 pos=");out_q16(a->R[R_POS_X]);out(",");out_q16(a->R[R_POS_Y]);out(",");out_q16(a->R[R_POS_Z]);
    out(" velX=");out_q16(a->R[R_VEL_X]);
    out(" latent=");out_q16(a->R[R_LATENT_VAL]);
    out(" handle=");out_i32(a->R[R_VOX_ADDR]);nl();
}

void start(void){
    g_out=GetStdHandle(STD_OUTPUT_HANDLE);
    out("DM-vOmegaXi+ 6400^3 VIRTUAL VRAM 3D SWARM ISA ENGINE\r\n");
    out("Windows x86-64 | 32-bit fixed-width ISA | Q16.16 arithmetic | 64 virtual agents\r\n");
    out("Virtual extent per axis: 6400 GiB | sparse pages: 64 KiB | resident slots: 2048\r\n");
    out("R10/R13/R14 are typed control registers; arithmetic registers are saturating Q16.16.\r\n");
    rule();print_program();rule();
    for(U32 i=0;i<ACTIVE_AGENTS;i++)init_agent(&A[i],i);

    while(g_sched_cycles<MAX_SCHED_CYCLES){
        for(U32 i=0;i<ACTIVE_AGENTS;i++)step_agent(&A[i]);
        release_barrier();
        if(all_halted())next_epoch();
        g_sched_cycles++;
        if((g_sched_cycles%256U)==0U)telemetry();
        Sleep(1);
    }
    rule();
    out("FINAL ");telemetry();
    out("logical sparse bytes materialized=");out_u64(g_logical_touched);nl();
    out("verify_failures=");out_u64(g_verify_fail);nl();
    out("HALT: bounded scheduler cycle budget reached.\r\n");
    ExitProcess(g_verify_fail?2U:0U);
}
