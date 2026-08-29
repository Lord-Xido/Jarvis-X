package com.engine.virtual.llm.operational;

import java.util.SplittableRandom;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.concurrent.atomic.AtomicLong;

/** Java 17 geometric 3D recurrent-field runtime with toroidal topology and fixed-point control. */
public final class OperationalInvertedLLMEngine implements AutoCloseable {
    public static final double TORUS_R1 = 1000.0;
    public static final double TORUS_R2 = 250.0;
    private static final double TWO_PI = 2.0 * Math.PI;
    private static final double L2 = 1.0e-5;
    private static final int CONVERGENCE_WINDOW = 3;

    private final ForkJoinPool pool;
    private final Volume volume;
    private final Telemetry telemetry = new Telemetry();

    public OperationalInvertedLLMEngine(int dim, int threads) { this(dim, threads, 41L); }
    public OperationalInvertedLLMEngine(int dim, int threads, long seed) {
        if (threads < 1) throw new IllegalArgumentException("threads must be >= 1");
        pool = new ForkJoinPool(threads);
        volume = new Volume(dim, seed);
    }

    public record GeometryPoint(float x, float y, float z, float curvature, float drive) {}
    public record CycleMetrics(long cycle, double fixedPointLoss, double fixedPointRms,
            double neighborCoherenceLoss, double meanParameterUpdate, double meanGradient,
            double stabilityPercent, long durationNanos) {}
    public record ConvergenceReport(boolean converged, int iterations, CycleMetrics finalMetrics,
            float feedbackCoupling, float learningRate, float neighborCoupling) {}

    public static final class Telemetry {
        private final AtomicLong cycles = new AtomicLong();
        private final DoubleAdder fixedLoss = new DoubleAdder();
        private final DoubleAdder updates = new DoubleAdder();
        private volatile CycleMetrics latest = new CycleMetrics(0, 0, Double.POSITIVE_INFINITY, 0, 0, 0, 0, 0);
        private void record(CycleMetrics m) { cycles.incrementAndGet(); fixedLoss.add(m.fixedPointLoss()); updates.add(m.meanParameterUpdate()); latest = m; }
        public long processedCycles() { return cycles.get(); }
        public CycleMetrics latest() { return latest; }
        public double meanFixedPointLoss() { long n=cycles.get(); return n==0?0:fixedLoss.doubleValue()/n; }
        public double meanParameterUpdate() { long n=cycles.get(); return n==0?0:updates.doubleValue()/n; }
    }

    private static final class Volume {
        final int dim, nodes;
        final float[] weights, activations, gradients, nextWeights, nextActivations;
        final float[] gx, gy, gz, curvature, drive;
        Volume(int dim, long seed) {
            if (dim < 3) throw new IllegalArgumentException("dim must be >= 3");
            long n;
            try { n = Math.multiplyExact(Math.multiplyExact((long)dim, dim), dim); }
            catch (ArithmeticException e) { throw new IllegalArgumentException("dimension overflow", e); }
            if (n > Integer.MAX_VALUE - 8L) throw new IllegalArgumentException("volume exceeds Java array capacity");
            this.dim=dim; this.nodes=(int)n;
            weights=new float[nodes]; activations=new float[nodes]; gradients=new float[nodes];
            nextWeights=new float[nodes]; nextActivations=new float[nodes];
            gx=new float[nodes]; gy=new float[nodes]; gz=new float[nodes]; curvature=new float[nodes]; drive=new float[nodes];
            SplittableRandom rnd=new SplittableRandom(seed);
            for(int z=0;z<dim;z++){
                double psi=TWO_PI*z/dim;
                for(int y=0;y<dim;y++){
                    double phi=TWO_PI*y/dim;
                    for(int x=0;x<dim;x++){
                        double theta=TWO_PI*x/dim;
                        int i=index(dim,x,y,z);
                        double r=TORUS_R2*(0.75+0.25*Math.cos(psi));
                        double radial=TORUS_R1+r*Math.cos(phi);
                        double X=radial*Math.cos(theta), Y=radial*Math.sin(theta), Z=r*Math.sin(phi)+0.25*TORUS_R2*Math.sin(psi);
                        gx[i]=(float)X; gy[i]=(float)Y; gz[i]=(float)Z;
                        double k=Math.cos(phi)/(r*Math.max(1.0e-9,radial));
                        curvature[i]=(float)Math.tanh(Math.abs(k)*1.0e6);
                        double radius=Math.sqrt(X*X+Y*Y+Z*Z);
                        drive[i]=(float)(0.55*Math.sin(theta+psi)+0.30*Math.cos(phi-psi)+0.15*Math.sin(radius/TORUS_R1));
                        weights[i]=(float)((rnd.nextDouble()-0.5)*0.05);
                        activations[i]=(float)Math.tanh(drive[i]+0.1*(rnd.nextDouble()-0.5));
                    }
                }
            }
        }
        GeometryPoint point(int x,int y,int z){ int i=indexChecked(dim,x,y,z); return new GeometryPoint(gx[i],gy[i],gz[i],curvature[i],drive[i]); }
    }

    public CycleMetrics executeOperationalCycle(float feedback, float learningRate, float neighborCoupling) {
        range("feedback",feedback,0,2); range("learningRate",learningRate,1e-8f,1); range("neighborCoupling",neighborCoupling,0,2);
        long start=System.nanoTime(); Acc a=new Acc();
        pool.invoke(new Step(volume,0,volume.dim,feedback,learningRate,neighborCoupling,a));
        System.arraycopy(volume.nextActivations,0,volume.activations,0,volume.nodes);
        System.arraycopy(volume.nextWeights,0,volume.weights,0,volume.nodes);
        double n=volume.nodes, loss=a.fixed.doubleValue()/n, rms=Math.sqrt(2*loss), coh=a.coherence.doubleValue()/n;
        double update=a.update.doubleValue()/n, grad=a.gradient.doubleValue()/n;
        double stability=100.0/(1.0+20.0*rms+5.0*coh);
        CycleMetrics m=new CycleMetrics(telemetry.processedCycles()+1,loss,rms,coh,update,grad,stability,System.nanoTime()-start);
        telemetry.record(m); return m;
    }

    public ConvergenceReport startAutonomousOptimizationLoop(int maxIterations, float targetStability, double tolerance) {
        if(maxIterations<1) throw new IllegalArgumentException("maxIterations must be >= 1");
        range("targetStability",targetStability,0,100);
        if(!Double.isFinite(tolerance)||tolerance<=0) throw new IllegalArgumentException("tolerance must be finite and > 0");
        System.out.printf("=== Inverted 3D Geometric Neural Field ===%nVolume: %d^3 = %,d nodes | Threads: %d | Topology: periodic 3-torus%n",volume.dim,volume.nodes,pool.getParallelism());
        float feedback=0.85f, lr=0.015f, neighbor=0.35f; int window=0; CycleMetrics last=telemetry.latest();
        for(int i=1;i<=maxIterations;i++){
            last=executeOperationalCycle(feedback,lr,neighbor);
            System.out.printf("Cycle %03d | %4d ms | Lfp %.8e | RMS %.8e | Lnbr %.8e | dW %.8e | Stable %.4f%%%n",i,last.durationNanos()/1_000_000,last.fixedPointLoss(),last.fixedPointRms(),last.neighborCoherenceLoss(),last.meanParameterUpdate(),last.stabilityPercent());
            boolean ok=last.fixedPointRms()<=tolerance && last.stabilityPercent()>=targetStability;
            window=ok?window+1:0;
            if(window>=CONVERGENCE_WINDOW){ System.out.println("=== FIXED_POINT_CONVERGED ==="); return new ConvergenceReport(true,i,last,feedback,lr,neighbor); }
            if(last.stabilityPercent()>=targetStability*0.90){ lr=Math.max(1e-5f,lr*0.94f); feedback=Math.min(1.25f,feedback*1.01f); }
            else if(last.fixedPointRms()>0.10) lr=Math.max(1e-5f,lr*0.98f);
            neighbor=clamp(neighbor*1.002f,0.10f,0.60f);
        }
        System.out.println("=== MAX_ITERATIONS_REACHED_WITHOUT_FIXED_POINT ===");
        return new ConvergenceReport(false,maxIterations,last,feedback,lr,neighbor);
    }

    public Telemetry telemetry(){return telemetry;}
    public int dimension(){return volume.dim;}
    public int totalNodes(){return volume.nodes;}
    public GeometryPoint geometryAt(int x,int y,int z){return volume.point(x,y,z);}
    @Override public void close(){pool.shutdown();}

    private static final class Acc { final DoubleAdder fixed=new DoubleAdder(), coherence=new DoubleAdder(), update=new DoubleAdder(), gradient=new DoubleAdder(); }
    private static final class Step extends RecursiveAction {
        private final Volume v; private final int zs,ze; private final float feedback,lr,neighbor; private final Acc a;
        Step(Volume v,int zs,int ze,float feedback,float lr,float neighbor,Acc a){this.v=v;this.zs=zs;this.ze=ze;this.feedback=feedback;this.lr=lr;this.neighbor=neighbor;this.a=a;}
        @Override protected void compute(){ if(ze-zs<=2){runSlices();return;} int mid=(zs+ze)>>>1; invokeAll(new Step(v,zs,mid,feedback,lr,neighbor,a),new Step(v,mid,ze,feedback,lr,neighbor,a)); }
        private void runSlices(){
            int d=v.dim; float[] act=v.activations,w=v.weights;
            for(int z=zs;z<ze;z++){int zm=wrap(z-1,d),zp=wrap(z+1,d); for(int y=0;y<d;y++){int ym=wrap(y-1,d),yp=wrap(y+1,d); for(int x=0;x<d;x++){
                int xm=wrap(x-1,d),xp=wrap(x+1,d),i=index(d,x,y,z); float ai=act[i],wi=w[i];
                double mean=(act[index(d,xm,y,z)]+act[index(d,xp,y,z)]+act[index(d,x,ym,z)]+act[index(d,x,yp,z)]+act[index(d,x,y,zm)]+act[index(d,x,y,zp)])/6.0;
                double geometryGain=0.5+0.5*v.curvature[i]; double u=wi*ai+neighbor*mean+feedback*geometryGain*v.drive[i];
                float next=(float)Math.tanh(u); double residual=next-ai; double g=residual*(1.0-next*next)*ai+L2*wi; float nw=(float)(wi-lr*g);
                v.nextActivations[i]=next; v.nextWeights[i]=nw; v.gradients[i]=(float)g;
                double cr=next-mean; a.fixed.add(0.5*residual*residual); a.coherence.add(0.5*cr*cr); a.update.add(Math.abs(nw-wi)); a.gradient.add(Math.abs(g));
            }}}
        }
    }

    private static int index(int d,int x,int y,int z){return (z*d+y)*d+x;}
    private static int indexChecked(int d,int x,int y,int z){if(x<0||x>=d||y<0||y>=d||z<0||z>=d)throw new IndexOutOfBoundsException("coordinate outside volume");return index(d,x,y,z);}
    private static int wrap(int c,int d){int v=c%d;return v<0?v+d:v;}
    private static void range(String name,float v,float lo,float hi){if(!Float.isFinite(v)||v<lo||v>hi)throw new IllegalArgumentException(name+" must be finite in ["+lo+", "+hi+"]");}
    private static float clamp(float v,float lo,float hi){return Math.max(lo,Math.min(hi,v));}

    public static void main(String[] args){
        int dim=args.length>0?Integer.parseInt(args[0]):128, iterations=args.length>1?Integer.parseInt(args[1]):25;
        int cores=Math.max(1,Runtime.getRuntime().availableProcessors());
        try(OperationalInvertedLLMEngine engine=new OperationalInvertedLLMEngine(dim,cores,41L)){
            GeometryPoint p=engine.geometryAt(0,0,0);
            System.out.printf("Geometry origin: (%.3f, %.3f, %.3f), curvature=%.6f, drive=%.6f%n",p.x(),p.y(),p.z(),p.curvature(),p.drive());
            ConvergenceReport r=engine.startAutonomousOptimizationLoop(iterations,92.0f,5.0e-3);
            System.out.printf("Final: converged=%s iterations=%d fixedPointRms=%.8e stability=%.4f%%%n",r.converged(),r.iterations(),r.finalMetrics().fixedPointRms(),r.finalMetrics().stabilityPercent());
        }
    }
}
