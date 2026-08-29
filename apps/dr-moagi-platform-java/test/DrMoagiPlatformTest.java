import java.util.List;

public final class DrMoagiPlatformTest {
  public static void main(String[] args) {
    testToroidalIndexing();
    testKineticStateIsFinite();
    testArithmeticSurface();
    testDataSurface();
    testJsonEscaping();
    testJsonExtraction();
    System.out.println("DrMoagiPlatformTest: PASS");
  }

  static void testToroidalIndexing() {
    check(DrMoagiPlatform.index(-1, 0, 0) == DrMoagiPlatform.index(3, 0, 0), "x wrap");
    check(DrMoagiPlatform.index(0, 4, 0) == DrMoagiPlatform.index(0, 0, 0), "y wrap");
    check(DrMoagiPlatform.index(0, 0, -1) == DrMoagiPlatform.index(0, 0, 3), "z wrap");
  }

  static void testKineticStateIsFinite() {
    var encoder = new DrMoagiPlatform.TurnEncoder(0.78, 0.42);
    var target = encoder.encode(List.of(
        new DrMoagiPlatform.Turn("user", "encode this state", DrMoagiPlatform.Surface.CHAT, java.time.Instant.EPOCH)));
    var dynamics = new DrMoagiPlatform.KineticDynamics(0.70, 0.16, 0.07, 0.40, 24, 0.12);
    double[] state = new double[DrMoagiPlatform.VOLUME];
    double firstLoss = dynamics.step(state, target).loss();
    double lastLoss = firstLoss;
    for (int i = 0; i < 24; i++) {
      var step = dynamics.step(state, target);
      state = step.state();
      lastLoss = step.loss();
      for (double v : state) check(Double.isFinite(v) && Math.abs(v) <= 1.0, "bounded state");
    }
    check(lastLoss < firstLoss, "kinetic relaxation should reduce loss for fixture");
  }

  static void testArithmeticSurface() {
    var parser = new DrMoagiPlatform.ExpressionParser("sqrt(81) + max(3, 5) * 2^3");
    check(Math.abs(parser.parse() - 49.0) < 1e-12, "arithmetic parser");
  }

  static void testDataSurface() {
    var adapter = new DrMoagiPlatform.DataAdapter();
    var telemetry = new DrMoagiPlatform.Telemetry(0, 0, 0, "TEST");
    var request = new DrMoagiPlatform.PlatformRequest(
        "1, 2, 3, 4", DrMoagiPlatform.Surface.DATA, List.of(), new double[DrMoagiPlatform.VOLUME], telemetry);
    String output = adapter.execute(request);
    check(output.contains("n=4") && output.contains("mean=2.50000000"), "data adapter");
  }

  static void testJsonEscaping() {
    String escaped = DrMoagiPlatform.jsonEscape("a\"b\\c\n");
    check(escaped.equals("a\\\"b\\\\c\\n"), "json escaping");
  }

  static void testJsonExtraction() {
    String json = "{\"choices\":[{\"message\":{\"content\":\"hello \\\"world\\\"\\nnext\"}}]}";
    String value = DrMoagiPlatform.extractJsonString(json, "content");
    check("hello \"world\"\nnext".equals(value), "json extraction");
  }

  static void check(boolean condition, String message) {
    if (!condition) throw new AssertionError(message);
  }
}
