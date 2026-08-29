package com.engine.virtual.llm.operational;

public final class OperationalInvertedLLMEngineTest {
    public static void main(String[] args) {
        testGeometryAndTopology();
        testOperationalCycleIsFinite();
        testBoundsValidation();
        System.out.println("OperationalInvertedLLMEngineTest: PASS");
    }

    private static void testGeometryAndTopology() {
        try (OperationalInvertedLLMEngine engine = new OperationalInvertedLLMEngine(8, 2, 7L)) {
            if (engine.totalNodes() != 512) throw new AssertionError("expected 512 nodes");
            var p = engine.geometryAt(0, 0, 0);
            if (!Float.isFinite(p.x()) || !Float.isFinite(p.y()) || !Float.isFinite(p.z())) {
                throw new AssertionError("geometry must be finite");
            }
        }
    }

    private static void testOperationalCycleIsFinite() {
        try (OperationalInvertedLLMEngine engine = new OperationalInvertedLLMEngine(10, 3, 11L)) {
            var m1 = engine.executeOperationalCycle(0.85f, 0.01f, 0.35f);
            var m2 = engine.executeOperationalCycle(0.85f, 0.01f, 0.35f);
            assertFinite(m1.fixedPointLoss());
            assertFinite(m1.fixedPointRms());
            assertFinite(m1.neighborCoherenceLoss());
            assertFinite(m1.stabilityPercent());
            assertFinite(m2.fixedPointLoss());
            if (engine.telemetry().processedCycles() != 2) throw new AssertionError("telemetry cycle count mismatch");
            if (m2.cycle() != 2) throw new AssertionError("cycle sequence mismatch");
        }
    }

    private static void testBoundsValidation() {
        boolean failed = false;
        try (OperationalInvertedLLMEngine ignored = new OperationalInvertedLLMEngine(2, 1, 1L)) {
            // unreachable
        } catch (IllegalArgumentException expected) {
            failed = true;
        }
        if (!failed) throw new AssertionError("dimension validation must reject dim < 3");
    }

    private static void assertFinite(double value) {
        if (!Double.isFinite(value)) throw new AssertionError("expected finite value, got " + value);
    }
}
