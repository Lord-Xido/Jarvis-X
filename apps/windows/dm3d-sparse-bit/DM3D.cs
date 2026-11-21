using System;
using System.IO;
using System.Text;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Security.Cryptography;

struct Cell : IEquatable<Cell>, IComparable<Cell> {
 public long X,Y,Z;
 public Cell(long x,long y,long z){X=x;Y=y;Z=z;}
 public bool Equals(Cell p){return X==p.X&&Y==p.Y&&Z==p.Z;}
 public override bool Equals(object o){return o is Cell&&Equals((Cell)o);}
 public override int GetHashCode(){unchecked{return (X.GetHashCode()*397^Y.GetHashCode())*397^Z.GetHashCode();}}
 public int CompareTo(Cell p){int c=X.CompareTo(p.X);if(c==0)c=Y.CompareTo(p.Y);return c==0?Z.CompareTo(p.Z):c;}
 public override string ToString(){return X+","+Y+","+Z;}
}
sealed class Machine {
 public const long DefaultAxis=8000000000000L;
 public long Axis=DefaultAxis;
 public uint Iteration;
 public const int MaxActive=100000, MaxIterations=100;
 public HashSet<Cell> Active=new HashSet<Cell>();
 static readonly int[,] D={{0,0,0},{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
 public bool Valid(Cell p){return p.X>=0&&p.Y>=0&&p.Z>=0&&p.X<Axis&&p.Y<Axis&&p.Z<Axis;}
 public void Check(Cell p){if(!Valid(p))throw new InvalidDataException("Coordinate outside axis bounds");}
 public int Read(Cell p){Check(p);return Active.Contains(p)?1:0;}
 public void Write(Cell p,int bit){Check(p);if(bit!=0&&bit!=1)throw new InvalidDataException("Bit must be 0 or 1");if(bit==0)Active.Remove(p);else {if(!Active.Contains(p)&&Active.Count>=MaxActive)throw new InvalidDataException("Active cell quota");Active.Add(p);}}
 public void Step(){
  if(Iteration>=MaxIterations)throw new InvalidDataException("Iteration quota");
  var watch=Stopwatch.StartNew();var next=new HashSet<Cell>();int n=0;
  // Toggle contributions in a new set: linear GF(2) convolution, then atomic commit.
  foreach(Cell p in Active){for(int j=0;j<7;j++){Cell q=new Cell(p.X+D[j,0],p.Y+D[j,1],p.Z+D[j,2]);if(Valid(q)&&!next.Add(q))next.Remove(q);}
   if(++n%1024==0&&watch.ElapsedMilliseconds>10000)throw new TimeoutException("STEP time quota; old state retained");}
  if(next.Count>MaxActive)throw new InvalidDataException("Active cell quota; old state retained");
  Active=next;Iteration++;
 }
 static void Put(Stream s,ulong v,int bytes){for(int i=bytes-1;i>=0;i--)s.WriteByte((byte)(v>>(8*i)));}
 static ulong Get(Stream s,int bytes){ulong v=0;for(int i=0;i<bytes;i++){int b=s.ReadByte();if(b<0)throw new EndOfStreamException();v=(v<<8)|(byte)b;}return v;}
 public byte[] Encode(){using(var s=new MemoryStream()){s.Write(new byte[]{68,77,51,68,1},0,5);Put(s,(ulong)Axis,6);Put(s,Iteration,4);Put(s,(ulong)Active.Count,8);foreach(Cell p in Active.OrderBy(x=>x)){Put(s,(ulong)p.X,6);Put(s,(ulong)p.Y,6);Put(s,(ulong)p.Z,6);}return s.ToArray();}}
 public static Machine Decode(byte[] b){
  if(b.Length<23||b[0]!=68||b[1]!=77||b[2]!=51||b[3]!=68||b[4]!=1)throw new InvalidDataException("Snapshot header");
  using(var s=new MemoryStream(b)){s.Position=5;long axis=(long)Get(s,6);uint iteration=(uint)Get(s,4);ulong count=Get(s,8);
   if(axis<1||axis>DefaultAxis||iteration>MaxIterations||count>MaxActive||b.Length!=23L+18L*(long)count)throw new InvalidDataException("Snapshot bounds or quota");
   var vm=new Machine{Axis=axis,Iteration=iteration};Cell last=new Cell();
   for(ulong i=0;i<count;i++){Cell p=new Cell((long)Get(s,6),(long)Get(s,6),(long)Get(s,6));vm.Check(p);if(i>0&&last.CompareTo(p)>=0)throw new InvalidDataException("Snapshot order or duplicate");vm.Active.Add(p);last=p;}return vm;
  }
 }
 public string Digest(){using(var sha=SHA256.Create())return BitConverter.ToString(sha.ComputeHash(Encode())).Replace("-","").ToLowerInvariant();}
 public BigInteger Morton(Cell p){Check(p);BigInteger m=0;for(int i=0;i<43;i++){m|=(BigInteger)((p.X>>i)&1)<<(3*i);m|=(BigInteger)((p.Y>>i)&1)<<(3*i+1);m|=(BigInteger)((p.Z>>i)&1)<<(3*i+2);}return m;}
 public Cell Unmorton(BigInteger m){if(m<0||m>=(BigInteger.One<<129))throw new InvalidDataException("Morton range");long[] a=new long[3];for(int i=0;i<43;i++)for(int j=0;j<3;j++)if(!((m>>(3*i+j))&1).IsZero)a[j]|=1L<<i;Cell p=new Cell(a[0],a[1],a[2]);Check(p);return p;}
}
sealed class Ins { public byte Op,A,B,C;public ulong X,Y,Z; }
static class Bytecode {
 public static List<Ins> Assemble(string text){var list=new List<Ins>();foreach(string raw in text.Split('\n')){
  string line=raw.Split('#')[0].Trim();if(line.Length==0)continue;string[] a=line.Replace(","," ").Split((char[])null,StringSplitOptions.RemoveEmptyEntries);var i=new Ins();
  switch(a[0].ToUpperInvariant()){
   case "LOAD":Need(a,5);i.Op=1;i.A=Reg(a[1]);Coords(i,a,2);break;
   case "STORE":Need(a,5);i.Op=2;Coords(i,a,1);i.A=Reg(a[4]);break;
   case "XOR":Need(a,4);i.Op=3;i.A=Reg(a[1]);i.B=Reg(a[2]);i.C=Reg(a[3]);break;
   case "MOV":Need(a,3);i.Op=4;i.A=Reg(a[1]);i.X=UInt64.Parse(a[2]);break;
   case "JNZ":Need(a,3);i.Op=5;i.A=Reg(a[1]);i.X=UInt64.Parse(a[2]);break;
   case "STEP":Need(a,1);i.Op=6;break;case "HASH":Need(a,1);i.Op=7;break;case "HALT":Need(a,1);i.Op=8;break;
   default:throw new InvalidDataException("Unknown instruction: "+a[0]);
  }list.Add(i);if(list.Count>100000)throw new InvalidDataException("Program length quota");
 }Verify(list,Machine.DefaultAxis);return list;}
 static void Need(string[] a,int n){if(a.Length!=n)throw new InvalidDataException("Wrong operand count");}
 static byte Reg(string s){if(s.Length<2||Char.ToUpperInvariant(s[0])!='R')throw new InvalidDataException("Expected R0 through R15");byte n=Byte.Parse(s.Substring(1));if(n>15)throw new InvalidDataException("Register range");return n;}
 static void Coords(Ins i,string[] a,int n){i.X=UInt64.Parse(a[n]);i.Y=UInt64.Parse(a[n+1]);i.Z=UInt64.Parse(a[n+2]);}
 public static void Verify(List<Ins> p,long axis){
  if(p.Count<1||p.Count>100000)throw new InvalidDataException("Program length");
  foreach(var i in p){if(i.Op<1||i.Op>8||i.A>15||i.B>15||i.C>15)throw new InvalidDataException("Opcode or register");
   if(i.Op==1||i.Op==2){if(i.B!=0||i.C!=0||i.X>=(ulong)axis||i.Y>=(ulong)axis||i.Z>=(ulong)axis)throw new InvalidDataException("Memory operands");}
   else if(i.Op==3){if(i.X!=0||i.Y!=0||i.Z!=0)throw new InvalidDataException("XOR reserved fields");}
   else if(i.Op==4||i.Op==5){if(i.B!=0||i.C!=0||i.Y!=0||i.Z!=0||(i.Op==4?i.X>1:i.X>=(ulong)p.Count))throw new InvalidDataException("MOV or JNZ operands");}
   else if(i.A!=0||i.B!=0||i.C!=0||i.X!=0||i.Y!=0||i.Z!=0)throw new InvalidDataException("Reserved fields");
  }
  var seen=new HashSet<int>();var todo=new Stack<int>();todo.Push(0);bool halt=false;
  while(todo.Count>0){int pc=todo.Pop();if(pc>=p.Count)throw new InvalidDataException("Reachable fallthrough beyond code");if(!seen.Add(pc))continue;var i=p[pc];if(i.Op==8){halt=true;continue;}todo.Push(pc+1);if(i.Op==5)todo.Push((int)i.X);}
  if(!halt)throw new InvalidDataException("No reachable HALT");
 }
 public static byte[] Encode(List<Ins> p){using(var m=new MemoryStream())using(var w=new BinaryWriter(m)){w.Write(new byte[]{68,51,86,77});w.Write(p.Count);foreach(var i in p){w.Write(i.Op);w.Write(i.A);w.Write(i.B);w.Write(i.C);w.Write(0U);w.Write(i.X);w.Write(i.Y);w.Write(i.Z);}return m.ToArray();}}
 public static List<Ins> Decode(byte[] b,long axis){using(var m=new MemoryStream(b))using(var r=new BinaryReader(m)){if(b.Length<8||r.ReadUInt32()!=0x4d563344)throw new InvalidDataException("Bytecode header");int n=r.ReadInt32();if(n<1||n>100000||b.Length!=8L+32L*n)throw new InvalidDataException("Bytecode length");var p=new List<Ins>();for(int j=0;j<n;j++){var i=new Ins{Op=r.ReadByte(),A=r.ReadByte(),B=r.ReadByte(),C=r.ReadByte()};if(r.ReadUInt32()!=0)throw new InvalidDataException("Reserved bytes");i.X=r.ReadUInt64();i.Y=r.ReadUInt64();i.Z=r.ReadUInt64();p.Add(i);}Verify(p,axis);return p;}}
 public static string Run(Machine vm,List<Ins> p,int budget){Verify(p,vm.Axis);int[] r=new int[16];int pc=0,steps=0;string digest="";var watch=Stopwatch.StartNew();
  while(true){if(++steps>budget)throw new InvalidDataException("Instruction quota");if(watch.ElapsedMilliseconds>10000)throw new TimeoutException("Program time quota");var i=p[pc];int next=pc+1;
   switch(i.Op){case 1:r[i.A]=vm.Read(new Cell((long)i.X,(long)i.Y,(long)i.Z));break;case 2:vm.Write(new Cell((long)i.X,(long)i.Y,(long)i.Z),r[i.A]);break;case 3:r[i.A]=r[i.B]^r[i.C];break;case 4:r[i.A]=(int)i.X;break;case 5:if(r[i.A]!=0)next=(int)i.X;break;case 6:vm.Step();break;case 7:digest=vm.Digest();break;case 8:return "HALT instructions="+steps+" active="+vm.Active.Count+" hash="+digest;}
   pc=next;
  }
 }
}
sealed class Router {
 public Machine VM=new Machine();readonly string root;readonly string audit;
 public Router(string folder){root=Path.GetFullPath(folder);Directory.CreateDirectory(root);if((File.GetAttributes(root)&FileAttributes.ReparsePoint)!=0)throw new IOException("Data directory cannot be a link");audit=Path.Combine(root,"audit.log");}
 string FileName(string name){if(name.Length<1||name.Length>80||name.Any(c=>!Char.IsLetterOrDigit(c)&&c!='.'&&c!='_'&&c!='-')||name=="."||name==".."||name.EndsWith("."))throw new InvalidDataException("Use a plain local file name");string p=Path.Combine(root,name);if(File.Exists(p)&&(File.GetAttributes(p)&FileAttributes.ReparsePoint)!=0)throw new IOException("Linked files are not allowed");if(String.Equals(p,audit,StringComparison.OrdinalIgnoreCase))throw new IOException("Audit file is reserved");return p;}
 byte[] ReadFile(string name,int max){string p=FileName(name);using(var f=new FileStream(p,FileMode.Open,FileAccess.Read,FileShare.Read)){if(f.Length>max)throw new InvalidDataException("File size quota");byte[] b=new byte[(int)f.Length];int n=0;while(n<b.Length){int k=f.Read(b,n,b.Length-n);if(k==0)throw new EndOfStreamException();n+=k;}return b;}}
 void Save(string name,byte[] b){string target=FileName(name),tmp=Path.Combine(root,Guid.NewGuid().ToString("N")+".tmp");var files=new DirectoryInfo(root).GetFiles();if(files.Length>=256||files.Sum(f=>f.Length)+(long)b.Length>268435456L)throw new IOException("Data folder quota");try{using(var f=new FileStream(tmp,FileMode.CreateNew,FileAccess.Write,FileShare.None)){f.Write(b,0,b.Length);f.Flush(true);}if(File.Exists(target))File.Replace(tmp,target,null);else File.Move(tmp,target);}finally{if(File.Exists(tmp))File.Delete(tmp);}}
 public string Execute(string command){
  string[] parts=command.Trim().Split((char[])null,StringSplitOptions.RemoveEmptyEntries);if(parts.Length==0)return "";
  if(parts[0].Equals("help",StringComparison.OrdinalIgnoreCase))return Dispatch(command);
  if(File.Exists(audit)&&(File.GetAttributes(audit)&FileAttributes.ReparsePoint)!=0)throw new IOException("Audit cannot be a link");
  if(File.Exists(audit)&&new FileInfo(audit).Length>4000000)throw new IOException("Audit quota; archive audit.log before further commands");
  string id=Guid.NewGuid().ToString("N"),op=parts[0].Length<=20&&parts[0].All(Char.IsLetter)?parts[0]:"invalid";
  Log(id+" BEGIN "+op);
  try{string result=Dispatch(command);Log(id+" OK active="+VM.Active.Count+" iteration="+VM.Iteration);return result;}
  catch(Exception e){Log(id+" FAILED "+e.GetType().Name);throw;}
 }
 void Log(string text){using(var f=new FileStream(audit,FileMode.Append,FileAccess.Write,FileShare.Read)){byte[] b=Encoding.UTF8.GetBytes(DateTime.UtcNow.ToString("o")+" "+text+Environment.NewLine);f.Write(b,0,b.Length);f.Flush(true);}}
 string Dispatch(string command){string[] a=command.Trim().Split((char[])null,StringSplitOptions.RemoveEmptyEntries);if(a.Length==0)return "";string op=a[0].ToLowerInvariant();string result;
  // Bounded local operation router. No external application or network interception.
  switch(op){
   case "help":return "status | reset | read x y z | write x y z bit | step [count] | hash | cells [limit]\nsave name | load name | assemble source.asm output.d3b | run program.d3b\nmorton x y z | unmorton integer | selftest | quit\nFiles are scoped to: "+root;
   case "status":Need(a,1);result="axis="+VM.Axis+" iteration="+VM.Iteration+" active="+VM.Active.Count;break;
   case "reset":Need(a,1);VM=new Machine();result="State reset";break;
   case "read":Need(a,4);result=VM.Read(Coord(a,1)).ToString();break;
   case "write":Need(a,5);VM.Write(Coord(a,1),Int32.Parse(a[4]));result="Stored";break;
   case "step":if(a.Length<1||a.Length>2)throw new InvalidDataException("step [count]");int count=a.Length==1?1:Int32.Parse(a[1]);if(count<1||count>100)throw new InvalidDataException("Step count 1 to 100");for(int j=0;j<count;j++)VM.Step();result="iteration="+VM.Iteration+" active="+VM.Active.Count;break;
   case "hash":Need(a,1);result=VM.Digest();break;
   case "cells":if(a.Length>2)throw new InvalidDataException("cells [limit]");int lim=a.Length==1?20:Int32.Parse(a[1]);if(lim<0||lim>1000)throw new InvalidDataException("Cell display limit");result=String.Join("\n",VM.Active.OrderBy(x=>x).Take(lim).Select(x=>x.ToString()));break;
   case "save":Need(a,2);Save(a[1],VM.Encode());result="Saved "+a[1];break;
   case "load":Need(a,2);VM=Machine.Decode(ReadFile(a[1],1800023));result="Loaded "+a[1];break;
   case "assemble":Need(a,3);var program=Bytecode.Assemble(new UTF8Encoding(false,true).GetString(ReadFile(a[1],4000000)));Save(a[2],Bytecode.Encode(program));result="Assembled "+program.Count+" instructions";break;
   case "run":Need(a,2);result=Bytecode.Run(VM,Bytecode.Decode(ReadFile(a[1],3200008),VM.Axis),1000000);break;
   case "morton":Need(a,4);result=VM.Morton(Coord(a,1)).ToString();break;
   case "unmorton":Need(a,2);result=VM.Unmorton(BigInteger.Parse(a[1])).ToString();break;
   case "selftest":Need(a,1);result=Tests.Run();break;
   default:throw new InvalidDataException("Unknown operation; type help");
  }
  return result;
 }
 static void Need(string[] a,int n){if(a.Length!=n)throw new InvalidDataException("Wrong argument count");}
 static Cell Coord(string[] a,int n){return new Cell(Int64.Parse(a[n]),Int64.Parse(a[n+1]),Int64.Parse(a[n+2]));}
}
static class Tests {
 static int checks;
 static void Assert(bool x,string name){checks++;if(!x)throw new Exception("SELFTEST FAIL: "+name);}
 static void Reject(Action a,string name){bool rejected=false;try{a();}catch(InvalidDataException){rejected=true;}Assert(rejected,name);}
 public static string Run(){checks=0;var vm=new Machine();vm.Write(new Cell(2,2,2),1);vm.Step();Assert(vm.Active.Count==7,"one hot");Assert(Machine.Decode(vm.Encode()).Digest()==vm.Digest(),"codec");
  Reject(()=>vm.Read(new Cell(-1,0,0)),"bounds");Reject(()=>vm.Write(new Cell(0,0,0),2),"bit");
  var random=new Random(312);for(int n=0;n<100;n++){var p=new Cell((long)(random.NextDouble()*Machine.DefaultAxis),(long)(random.NextDouble()*Machine.DefaultAxis),(long)(random.NextDouble()*Machine.DefaultAxis));Assert(vm.Unmorton(vm.Morton(p)).Equals(p),"Morton");}
  for(int trial=0;trial<8;trial++){var small=new Machine{Axis=5};bool[,,] dense=new bool[5,5,5];for(int x=0;x<5;x++)for(int y=0;y<5;y++)for(int z=0;z<5;z++)if(random.Next(4)==0){dense[x,y,z]=true;small.Write(new Cell(x,y,z),1);}
   for(int t=0;t<3;t++){var next=new bool[5,5,5];for(int x=0;x<5;x++)for(int y=0;y<5;y++)for(int z=0;z<5;z++){bool v=dense[x,y,z];if(x>0)v^=dense[x-1,y,z];if(x<4)v^=dense[x+1,y,z];if(y>0)v^=dense[x,y-1,z];if(y<4)v^=dense[x,y+1,z];if(z>0)v^=dense[x,y,z-1];if(z<4)v^=dense[x,y,z+1];next[x,y,z]=v;}small.Step();dense=next;for(int x=0;x<5;x++)for(int y=0;y<5;y++)for(int z=0;z<5;z++)Assert(small.Read(new Cell(x,y,z))==(dense[x,y,z]?1:0),"dense reference");}
  }
  var p1=Bytecode.Assemble("MOV R0 1\nSTORE 2 2 2 R0\nSTEP\nHASH\nHALT");var run=new Machine();Bytecode.Run(run,Bytecode.Decode(Bytecode.Encode(p1),run.Axis),20);Assert(run.Active.Count==7,"bytecode execution");
  Reject(()=>Bytecode.Assemble("JNZ R0 99\nHALT"),"branch bounds");Reject(()=>Bytecode.Assemble("MOV R0 1"),"fallthrough");
  Reject(()=>Bytecode.Run(new Machine(),Bytecode.Assemble("MOV R0 1\nJNZ R0 1\nHALT"),20),"loop quota");
  var corrupt=vm.Encode();corrupt[0]=0;Reject(()=>Machine.Decode(corrupt),"bad magic");var duplicate=new Machine();duplicate.Write(new Cell(0,0,0),1);duplicate.Write(new Cell(1,1,1),1);var b=duplicate.Encode();Array.Copy(b,23,b,41,18);Reject(()=>Machine.Decode(b),"duplicate");
  var limited=new Machine{Iteration=100};string before=limited.Digest();Reject(()=>limited.Step(),"iteration quota");Assert(before==limited.Digest(),"atomic failure");
  return "PASS "+checks+" checks: dense parity, codec, bounds, Morton, bytecode, quotas";
 }
}
static class Program {
 static int Main(string[] args){try{
  if(args.Length==1&&args[0]=="--self-test"){Console.WriteLine(Tests.Run());return 0;}
  string root=Path.Combine(AppDomain.CurrentDomain.BaseDirectory,"DM3D_Data");var router=new Router(root);
  if(args.Length==2&&args[0]=="--batch"){var info=new FileInfo(args[1]);if(info.Length>1000000)throw new InvalidDataException("Batch file quota");int lines=0;foreach(string raw in File.ReadLines(args[1])){if(++lines>10000)throw new InvalidDataException("Batch line quota");string line=raw.Trim();if(line.Length==0||line.StartsWith("#"))continue;Console.WriteLine("> "+line);Console.WriteLine(router.Execute(line));}return 0;}
  if(args.Length>0){Console.WriteLine(router.Execute(String.Join(" ",args)));return 0;}
  Console.WriteLine("DM3D / JARVIS-X local virtual computer\nSparse bit VM and local operation router. Type help.\nNo external applications are redirected.");
  while(true){Console.Write("dm3d> ");string line=Console.ReadLine();if(line==null||line.Trim().Equals("quit",StringComparison.OrdinalIgnoreCase))break;try{Console.WriteLine(router.Execute(line));}catch(Exception e){Console.WriteLine("ERROR: "+e.Message);}}
  return 0;
 }catch(Exception e){Console.Error.WriteLine("ERROR: "+e.Message);return 1;}}
}
