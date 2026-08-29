import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Scanner;
import java.util.ServiceLoader;

/**
 * Dr Moagi Platform v0.5.0
 *
 * A dependency-free Java 17 multimodal computing kernel.
 *
 * Pipeline:
 *   request -> turn-wise encoder -> 3D toroidal latent field
 *   -> recurrent kinetic relaxation -> capability router
 *   -> chat/image/audio/video/code/data/compute surface
 *   -> telemetry -> recurrent transcript state
 *
 * External providers are optional. The local runtime remains useful offline.
 */
public final class DrMoagiPlatform {
  static final int SIDE = 4;
  static final int VOLUME = SIDE * SIDE * SIDE;
  static final double CONVERGENCE_EPSILON = 0.002;

  private DrMoagiPlatform() {}

  public static void main(String[] args) {
    Platform platform = Platform.bootstrap();
    System.out.println("DR MOAGI / GENERAL MULTIMODAL PLATFORM v0.5.0");
    System.out.println("Java 17 | latent=4x4x4 torus | dependency-free");
    System.out.println("Commands: /chat /image /audio /video /code /data /compute /status /reset /help /exit\n");

    try (Scanner scanner = new Scanner(new BufferedReader(new InputStreamReader(System.in)))) {
      while (true) {
        System.out.print("you> ");
        if (!scanner.hasNextLine()) break;
        String line = scanner.nextLine().trim();
        if (line.isBlank()) continue;
        if (line.equalsIgnoreCase("/exit")) break;
        if (line.equalsIgnoreCase("/help")) {
          System.out.println(helpText());
          continue;
        }
        if (line.equalsIgnoreCase("/status")) {
          System.out.println(platform.status());
          continue;
        }
        if (line.equalsIgnoreCase("/reset")) {
          platform.reset();
          System.out.println("moagi> State reset.\n");
          continue;
        }

        ParsedInput input = ParsedInput.parse(line);
        PlatformResponse response = platform.process(input.prompt(), input.surface());
        System.out.println("\nmoagi> " + response.output());
        System.out.printf(Locale.ROOT,
            "      signature=%s loss=%.6f steps=%d energy=%.6f mode=%s%n%n",
            response.telemetry().signature(),
            response.telemetry().loss(),
            response.telemetry().steps(),
            response.telemetry().energy(),
            response.surface().label);
      }
    }
  }

  static String helpText() {
    return """
        Surfaces
          /chat <prompt>       conversational reasoning surface
          /image <prompt>      image renderer adapter surface
          /audio <prompt>      audio renderer adapter surface
          /video <prompt>      video renderer adapter surface
          /code <prompt>       code-generation/review surface
          /data <numbers>      local numeric analytics surface
          /compute <expr>      safe local arithmetic surface

        Control
          /status              kernel/capability telemetry
          /reset               clear transcript and latent state
          /help                show this help
          /exit                stop the runtime

        External model configuration (optional)
          MOAGI_TEXT_ENDPOINT  OpenAI-compatible chat-completions endpoint
          MOAGI_API_KEY        bearer token
          MOAGI_MODEL          model identifier

        External media endpoint configuration (optional)
          MOAGI_IMAGE_ENDPOINT
          MOAGI_AUDIO_ENDPOINT
          MOAGI_VIDEO_ENDPOINT

        Media endpoints receive JSON: {mode,prompt,latent,signature} and may return any UTF-8 body.
        """;
  }

  enum Surface {
    CHAT("chat"), IMAGE("image"), AUDIO("audio"), VIDEO("video"),
    CODE("code"), DATA("data"), COMPUTE("compute");

    final String label;
    Surface(String label) { this.label = label; }

    static Surface fromCommand(String command) {
      return switch (command.toLowerCase(Locale.ROOT)) {
        case "/image" -> IMAGE;
        case "/audio" -> AUDIO;
        case "/video" -> VIDEO;
        case "/code" -> CODE;
        case "/data" -> DATA;
        case "/compute" -> COMPUTE;
        default -> CHAT;
      };
    }
  }

  record ParsedInput(Surface surface, String prompt) {
    static ParsedInput parse(String line) {
      if (!line.startsWith("/")) return new ParsedInput(Surface.CHAT, line);
      int split = line.indexOf(' ');
      String command = split < 0 ? line : line.substring(0, split);
      Surface surface = Surface.fromCommand(command);
      String prompt = split < 0 ? "Describe the current state." : line.substring(split + 1).trim();
      return new ParsedInput(surface, prompt.isBlank() ? "Describe the current state." : prompt);
    }
  }

  record Turn(String role, String content, Surface surface, Instant timestamp) {}

  record Telemetry(double loss, int steps, double energy, String signature) {}

  record PlatformRequest(
      String prompt,
      Surface surface,
      List<Turn> transcript,
      double[] latent,
      Telemetry telemetry) {
    PlatformRequest {
      latent = latent.clone();
      transcript = List.copyOf(transcript);
    }
  }

  record PlatformResponse(String output, Surface surface, Telemetry telemetry) {}

  public interface SurfaceAdapter {
    Surface surface();
    String execute(PlatformRequest request);
    default String name() { return getClass().getSimpleName(); }
  }

  interface LanguageModel {
    String complete(PlatformRequest request, String instruction);
    String name();
  }

  static final class Platform {
    private final TurnEncoder encoder;
    private final KineticDynamics dynamics;
    private final CapabilityRegistry registry;
    private final List<Turn> transcript = new ArrayList<>();
    private double[] latent = new double[VOLUME];

    Platform(TurnEncoder encoder, KineticDynamics dynamics, CapabilityRegistry registry) {
      this.encoder = Objects.requireNonNull(encoder);
      this.dynamics = Objects.requireNonNull(dynamics);
      this.registry = Objects.requireNonNull(registry);
    }

    static Platform bootstrap() {
      LanguageModel languageModel = createLanguageModel(System.getenv());
      CapabilityRegistry registry = new CapabilityRegistry();
      registry.register(new ChatAdapter(languageModel));
      registry.register(new CodeAdapter(languageModel));
      registry.register(new DataAdapter());
      registry.register(new ComputeAdapter());
      registry.register(mediaAdapter(Surface.IMAGE, "MOAGI_IMAGE_ENDPOINT"));
      registry.register(mediaAdapter(Surface.AUDIO, "MOAGI_AUDIO_ENDPOINT"));
      registry.register(mediaAdapter(Surface.VIDEO, "MOAGI_VIDEO_ENDPOINT"));
      ServiceLoader.load(SurfaceAdapter.class).forEach(registry::register);

      return new Platform(
          new TurnEncoder(0.78, 0.42),
          new KineticDynamics(0.70, 0.16, 0.07, 0.40, 24, 0.12),
          registry);
    }

    static LanguageModel createLanguageModel(Map<String, String> env) {
      String endpoint = env.getOrDefault("MOAGI_TEXT_ENDPOINT", "").trim();
      String apiKey = env.getOrDefault("MOAGI_API_KEY", "").trim();
      String model = env.getOrDefault("MOAGI_MODEL", "").trim();
      if (!endpoint.isBlank() && !apiKey.isBlank() && !model.isBlank()) {
        return new OpenAICompatibleLanguageModel(endpoint, apiKey, model);
      }
      return new LocalLanguageModel();
    }

    static SurfaceAdapter mediaAdapter(Surface surface, String envKey) {
      String endpoint = System.getenv().getOrDefault(envKey, "").trim();
      return endpoint.isBlank()
          ? new SpecMediaAdapter(surface)
          : new HttpMediaAdapter(surface, endpoint, System.getenv().getOrDefault("MOAGI_API_KEY", ""));
    }

    synchronized PlatformResponse process(String prompt, Surface surface) {
      Objects.requireNonNull(prompt);
      Objects.requireNonNull(surface);
      transcript.add(new Turn("user", prompt, surface, Instant.now()));

      double[] target = encoder.encode(transcript);
      double loss = Double.POSITIVE_INFINITY;
      int steps = 0;
      for (int i = 0; i < dynamics.maxSteps; i++) {
        KineticStep step = dynamics.step(latent, target);
        latent = step.state();
        loss = step.loss();
        steps++;
        if (loss < CONVERGENCE_EPSILON) break;
      }

      Telemetry telemetry = telemetry(loss, steps, latent);
      PlatformRequest request = new PlatformRequest(prompt, surface, transcript, latent, telemetry);
      String output = registry.resolve(surface).execute(request);
      transcript.add(new Turn("assistant", output, surface, Instant.now()));
      return new PlatformResponse(output, surface, telemetry);
    }

    synchronized void reset() {
      transcript.clear();
      latent = new double[VOLUME];
    }

    synchronized String status() {
      Telemetry t = telemetry(0.0, 0, latent);
      return """
          DR MOAGI PLATFORM STATUS
            version: 0.5.0
            java: %s
            latent-field: %dx%dx%d (%d cells, toroidal)
            transcript-turns: %d
            energy: %.6f
            signature: %s
            capabilities: %s
          """.formatted(
          System.getProperty("java.version"), SIDE, SIDE, SIDE, VOLUME,
          transcript.size(), t.energy(), t.signature(), registry.describe());
    }
  }

  /** Turn-wise recurrent encoder. Each turn contributes its own spatial field. */
  static final class TurnEncoder {
    private final double decay;
    private final double gain;

    TurnEncoder(double decay, double gain) {
      this.decay = decay;
      this.gain = gain;
    }

    double[] encode(List<Turn> transcript) {
      double[] state = new double[VOLUME];
      for (Turn turn : transcript) {
        double[] features = encodeTurn(turn);
        for (int i = 0; i < VOLUME; i++) {
          state[i] = Math.tanh(decay * state[i] + gain * features[i]);
        }
      }
      return state;
    }

    private double[] encodeTurn(Turn turn) {
      double[] field = new double[VOLUME];
      String text = turn.content();
      int rolling = 0x4d4f4147 ^ turn.surface().ordinal() * 0x9e3779b9;
      int words = text.isBlank() ? 0 : text.trim().split("\\s+").length;
      int vowels = 0;

      for (int i = 0; i < text.length(); i++) {
        char c = Character.toLowerCase(text.charAt(i));
        if ("aeiou".indexOf(c) >= 0) vowels++;
        rolling = 31 * rolling + c + i * 17;
        int index = Math.floorMod(rolling, VOLUME);
        double amplitude = ((c % 97) + 1) / 98.0;
        field[index] += amplitude;
        field[neighborIndex(index, 1, 0, 0)] += amplitude * 0.25;
        field[neighborIndex(index, 0, 1, 0)] -= amplitude * 0.12;
        field[neighborIndex(index, 0, 0, 1)] += amplitude * 0.08;
      }

      field[0] += Math.min(1.0, words / 64.0);
      field[1] += Math.min(1.0, vowels / 64.0);
      field[2] += turn.surface().ordinal() / (double) Math.max(1, Surface.values().length - 1);
      field[3] += "assistant".equals(turn.role()) ? -0.35 : 0.35;

      double max = 1.0;
      for (double value : field) max = Math.max(max, Math.abs(value));
      for (int i = 0; i < VOLUME; i++) field[i] = Math.tanh(field[i] / max * 2.0);
      return field;
    }
  }

  record KineticStep(double[] state, double loss) {
    KineticStep { state = state.clone(); }
  }

  /** Euler integration on a periodic 3D field; boundaries fold back into the field. */
  static final class KineticDynamics {
    final double alpha, beta, gamma, eta, deltaT;
    final int maxSteps;

    KineticDynamics(double alpha, double beta, double gamma, double eta, int maxSteps, double deltaT) {
      this.alpha = alpha;
      this.beta = beta;
      this.gamma = gamma;
      this.eta = eta;
      this.maxSteps = maxSteps;
      this.deltaT = deltaT;
    }

    KineticStep step(double[] state, double[] target) {
      if (state.length != VOLUME || target.length != VOLUME) {
        throw new IllegalArgumentException("latent fields must contain " + VOLUME + " cells");
      }
      double[] next = new double[VOLUME];
      for (int i = 0; i < VOLUME; i++) {
        double neighbor = neighborMean(state, i);
        double decoded = Math.tanh(state[i] + beta * (neighbor - state[i]));
        double error = target[i] - decoded;
        double force = alpha * (target[i] - state[i])
            + beta * (neighbor - state[i])
            - gamma * state[i]
            + eta * error;
        next[i] = clamp(state[i] + deltaT * force, -1.0, 1.0);
      }

      // Loss is evaluated on the authoritative next state, not the pre-update state.
      double loss = 0.0;
      for (int i = 0; i < VOLUME; i++) {
        double neighbor = neighborMean(next, i);
        double decoded = Math.tanh(next[i] + beta * (neighbor - next[i]));
        double error = target[i] - decoded;
        loss += error * error;
      }
      return new KineticStep(next, loss / VOLUME);
    }
  }

  static final class CapabilityRegistry {
    private final EnumMap<Surface, SurfaceAdapter> adapters = new EnumMap<>(Surface.class);

    synchronized void register(SurfaceAdapter adapter) {
      adapters.put(adapter.surface(), Objects.requireNonNull(adapter));
    }

    synchronized SurfaceAdapter resolve(Surface surface) {
      SurfaceAdapter adapter = adapters.get(surface);
      if (adapter == null) throw new IllegalStateException("No adapter registered for " + surface.label);
      return adapter;
    }

    synchronized String describe() {
      return adapters.entrySet().stream()
          .map(e -> e.getKey().label + "=" + e.getValue().name())
          .sorted()
          .reduce((a, b) -> a + ", " + b)
          .orElse("none");
    }
  }

  static final class ChatAdapter implements SurfaceAdapter {
    private final LanguageModel model;
    ChatAdapter(LanguageModel model) { this.model = model; }
    public Surface surface() { return Surface.CHAT; }
    public String execute(PlatformRequest request) {
      return model.complete(request,
          "Answer the user directly. Treat latent telemetry as state context, not as a claim of intelligence.");
    }
    public String name() { return "ChatAdapter(" + model.name() + ")"; }
  }

  static final class CodeAdapter implements SurfaceAdapter {
    private final LanguageModel model;
    CodeAdapter(LanguageModel model) { this.model = model; }
    public Surface surface() { return Surface.CODE; }
    public String execute(PlatformRequest request) {
      return model.complete(request,
          "Act as a software engineering surface. Produce precise implementation-oriented output, tests, and assumptions.");
    }
    public String name() { return "CodeAdapter(" + model.name() + ")"; }
  }

  static final class LocalLanguageModel implements LanguageModel {
    public String complete(PlatformRequest request, String instruction) {
      return "[LOCAL " + request.surface().label.toUpperCase(Locale.ROOT) + "] "
          + "Request accepted through latent " + request.telemetry().signature() + ". "
          + "No external language model is configured, so the dependency-free kernel is returning a deterministic "
          + "runtime acknowledgement for: “" + request.prompt() + "”.";
    }
    public String name() { return "local"; }
  }

  /** Optional OpenAI-compatible chat-completions adapter. */
  static final class OpenAICompatibleLanguageModel implements LanguageModel {
    private final HttpClient client = HttpClient.newBuilder().build();
    private final URI endpoint;
    private final String apiKey;
    private final String model;

    OpenAICompatibleLanguageModel(String endpoint, String apiKey, String model) {
      this.endpoint = URI.create(endpoint);
      this.apiKey = apiKey;
      this.model = model;
    }

    public String complete(PlatformRequest request, String instruction) {
      String latent = compactLatent(request.latent(), 12);
      String system = instruction + " Kernel signature=" + request.telemetry().signature()
          + "; latent-sample=" + latent + ".";
      String body = "{\"model\":\"" + jsonEscape(model) + "\",\"messages\":["
          + "{\"role\":\"system\",\"content\":\"" + jsonEscape(system) + "\"},"
          + "{\"role\":\"user\",\"content\":\"" + jsonEscape(request.prompt()) + "\"}]}";
      HttpRequest httpRequest = HttpRequest.newBuilder(endpoint)
          .header("Authorization", "Bearer " + apiKey)
          .header("Content-Type", "application/json")
          .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8))
          .build();
      try {
        HttpResponse<String> response = client.send(httpRequest, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
        if (response.statusCode() < 200 || response.statusCode() >= 300) {
          return "Provider HTTP " + response.statusCode() + ": " + trimBody(response.body());
        }
        String content = extractJsonString(response.body(), "content");
        return content == null ? "Provider returned no decodable text content: " + trimBody(response.body()) : content;
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        return "Provider request interrupted.";
      } catch (IOException | IllegalArgumentException e) {
        return "Provider error: " + e.getMessage();
      }
    }

    public String name() { return "openai-compatible:" + model; }
  }

  static final class SpecMediaAdapter implements SurfaceAdapter {
    private final Surface surface;
    SpecMediaAdapter(Surface surface) { this.surface = surface; }
    public Surface surface() { return surface; }
    public String execute(PlatformRequest request) {
      return "[" + surface.label.toUpperCase(Locale.ROOT) + " SPEC READY] "
          + "signature=" + request.telemetry().signature()
          + " prompt=“" + request.prompt() + "”. Configure MOAGI_"
          + surface.label.toUpperCase(Locale.ROOT) + "_ENDPOINT to attach a renderer.";
    }
  }

  /** Generic JSON media bridge owned by the platform contract. */
  static final class HttpMediaAdapter implements SurfaceAdapter {
    private final HttpClient client = HttpClient.newHttpClient();
    private final Surface surface;
    private final URI endpoint;
    private final String apiKey;

    HttpMediaAdapter(Surface surface, String endpoint, String apiKey) {
      this.surface = surface;
      this.endpoint = URI.create(endpoint);
      this.apiKey = apiKey == null ? "" : apiKey;
    }

    public Surface surface() { return surface; }

    public String execute(PlatformRequest request) {
      String body = "{\"mode\":\"" + surface.label + "\",\"prompt\":\""
          + jsonEscape(request.prompt()) + "\",\"latent\":" + latentJson(request.latent())
          + ",\"signature\":\"" + jsonEscape(request.telemetry().signature()) + "\"}";
      HttpRequest.Builder builder = HttpRequest.newBuilder(endpoint)
          .header("Content-Type", "application/json")
          .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8));
      if (!apiKey.isBlank()) builder.header("Authorization", "Bearer " + apiKey);
      try {
        HttpResponse<String> response = client.send(builder.build(), HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
        return "[" + surface.label.toUpperCase(Locale.ROOT) + " HTTP " + response.statusCode() + "] "
            + trimBody(response.body());
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        return "Media request interrupted.";
      } catch (IOException e) {
        return "Media provider error: " + e.getMessage();
      }
    }
  }

  static final class DataAdapter implements SurfaceAdapter {
    public Surface surface() { return Surface.DATA; }

    public String execute(PlatformRequest request) {
      String[] tokens = request.prompt().trim().split("[\\s,;]+");
      List<Double> values = new ArrayList<>();
      for (String token : tokens) {
        if (token.isBlank()) continue;
        try {
          values.add(Double.parseDouble(token));
        } catch (NumberFormatException ignored) {
          // Non-numeric tokens are intentionally skipped in the local analytics surface.
        }
      }
      if (values.isEmpty()) return "[DATA] No numeric values found.";
      double sum = values.stream().mapToDouble(Double::doubleValue).sum();
      double mean = sum / values.size();
      double min = values.stream().mapToDouble(Double::doubleValue).min().orElse(Double.NaN);
      double max = values.stream().mapToDouble(Double::doubleValue).max().orElse(Double.NaN);
      double variance = values.stream().mapToDouble(v -> (v - mean) * (v - mean)).sum() / values.size();
      return String.format(Locale.ROOT,
          "[DATA] n=%d sum=%.8f mean=%.8f min=%.8f max=%.8f variance=%.8f signature=%s",
          values.size(), sum, mean, min, max, variance, request.telemetry().signature());
    }
  }

  static final class ComputeAdapter implements SurfaceAdapter {
    public Surface surface() { return Surface.COMPUTE; }

    public String execute(PlatformRequest request) {
      try {
        double result = new ExpressionParser(request.prompt()).parse();
        return "[COMPUTE] " + request.prompt() + " = " + formatNumber(result)
            + " signature=" + request.telemetry().signature();
      } catch (RuntimeException e) {
        return "[COMPUTE ERROR] " + e.getMessage()
            + ". Supported: + - * / % ^ parentheses and sin cos tan sqrt abs log exp min max.";
      }
    }
  }

  /** Tiny safe arithmetic parser: no reflection, shell, filesystem, or process execution. */
  static final class ExpressionParser {
    private final String text;
    private int pos;

    ExpressionParser(String text) { this.text = text == null ? "" : text; }

    double parse() {
      double value = expression();
      skip();
      if (pos != text.length()) throw error("Unexpected token at position " + pos);
      if (!Double.isFinite(value)) throw error("Result is not finite");
      return value;
    }

    private double expression() {
      double value = term();
      while (true) {
        skip();
        if (eat('+')) value += term();
        else if (eat('-')) value -= term();
        else return value;
      }
    }

    private double term() {
      double value = power();
      while (true) {
        skip();
        if (eat('*')) value *= power();
        else if (eat('/')) value /= power();
        else if (eat('%')) value %= power();
        else return value;
      }
    }

    private double power() {
      double value = unary();
      skip();
      if (eat('^')) value = Math.pow(value, power());
      return value;
    }

    private double unary() {
      skip();
      if (eat('+')) return unary();
      if (eat('-')) return -unary();
      return primary();
    }

    private double primary() {
      skip();
      if (eat('(')) {
        double value = expression();
        require(')');
        return value;
      }
      if (pos < text.length() && Character.isLetter(text.charAt(pos))) {
        String name = identifier().toLowerCase(Locale.ROOT);
        if (name.equals("pi")) return Math.PI;
        if (name.equals("e")) return Math.E;
        require('(');
        double a = expression();
        skip();
        Double b = null;
        if (eat(',')) b = expression();
        require(')');
        return function(name, a, b);
      }
      return number();
    }

    private double function(String name, double a, Double b) {
      return switch (name) {
        case "sin" -> Math.sin(a);
        case "cos" -> Math.cos(a);
        case "tan" -> Math.tan(a);
        case "sqrt" -> Math.sqrt(a);
        case "abs" -> Math.abs(a);
        case "log" -> Math.log(a);
        case "exp" -> Math.exp(a);
        case "min" -> Math.min(a, requireSecond(name, b));
        case "max" -> Math.max(a, requireSecond(name, b));
        default -> throw error("Unknown function " + name);
      };
    }

    private static double requireSecond(String name, Double value) {
      if (value == null) throw new IllegalArgumentException(name + " requires two arguments");
      return value;
    }

    private String identifier() {
      int start = pos;
      while (pos < text.length() && Character.isLetter(text.charAt(pos))) pos++;
      return text.substring(start, pos);
    }

    private double number() {
      skip();
      int start = pos;
      boolean dot = false;
      boolean exponent = false;
      while (pos < text.length()) {
        char c = text.charAt(pos);
        if (Character.isDigit(c)) {
          pos++;
        } else if (c == '.' && !dot && !exponent) {
          dot = true;
          pos++;
        } else if ((c == 'e' || c == 'E') && !exponent) {
          exponent = true;
          pos++;
          if (pos < text.length() && (text.charAt(pos) == '+' || text.charAt(pos) == '-')) pos++;
        } else break;
      }
      if (start == pos) throw error("Expected number at position " + pos);
      return Double.parseDouble(text.substring(start, pos));
    }

    private void require(char c) {
      skip();
      if (!eat(c)) throw error("Expected '" + c + "' at position " + pos);
    }

    private boolean eat(char c) {
      if (pos < text.length() && text.charAt(pos) == c) {
        pos++;
        return true;
      }
      return false;
    }

    private void skip() {
      while (pos < text.length() && Character.isWhitespace(text.charAt(pos))) pos++;
    }

    private IllegalArgumentException error(String message) { return new IllegalArgumentException(message); }
  }

  static int index(int x, int y, int z) {
    int xx = Math.floorMod(x, SIDE);
    int yy = Math.floorMod(y, SIDE);
    int zz = Math.floorMod(z, SIDE);
    return (zz * SIDE + yy) * SIDE + xx;
  }

  static int neighborIndex(int i, int dx, int dy, int dz) {
    int x = i % SIDE;
    int y = (i / SIDE) % SIDE;
    int z = i / (SIDE * SIDE);
    return index(x + dx, y + dy, z + dz);
  }

  static double neighborMean(double[] field, int i) {
    return (field[neighborIndex(i, 1, 0, 0)] + field[neighborIndex(i, -1, 0, 0)]
        + field[neighborIndex(i, 0, 1, 0)] + field[neighborIndex(i, 0, -1, 0)]
        + field[neighborIndex(i, 0, 0, 1)] + field[neighborIndex(i, 0, 0, -1)]) / 6.0;
  }

  static Telemetry telemetry(double loss, int steps, double[] latent) {
    double energy = Arrays.stream(latent).map(v -> v * v).average().orElse(0.0);
    return new Telemetry(loss, steps, energy, signature(latent));
  }

  static String signature(double[] latent) {
    long hash = 0xcbf29ce484222325L;
    for (double value : latent) {
      long bits = Double.doubleToLongBits(Math.rint(value * 1_000_000.0) / 1_000_000.0);
      hash ^= bits;
      hash *= 0x100000001b3L;
    }
    return "DM3D-" + Long.toUnsignedString(hash, 16).toUpperCase(Locale.ROOT);
  }

  static String compactLatent(double[] latent, int count) {
    StringBuilder out = new StringBuilder("[");
    for (int i = 0; i < Math.min(count, latent.length); i++) {
      if (i > 0) out.append(',');
      out.append(String.format(Locale.ROOT, "%.4f", latent[i]));
    }
    if (latent.length > count) out.append(",…");
    return out.append(']').toString();
  }

  static String latentJson(double[] latent) {
    StringBuilder out = new StringBuilder("[");
    for (int i = 0; i < latent.length; i++) {
      if (i > 0) out.append(',');
      out.append(String.format(Locale.ROOT, "%.8f", latent[i]));
    }
    return out.append(']').toString();
  }

  static String jsonEscape(String value) {
    StringBuilder out = new StringBuilder(value.length() + 16);
    for (int i = 0; i < value.length(); i++) {
      char c = value.charAt(i);
      switch (c) {
        case '"' -> out.append("\\\"");
        case '\\' -> out.append("\\\\");
        case '\b' -> out.append("\\b");
        case '\f' -> out.append("\\f");
        case '\n' -> out.append("\\n");
        case '\r' -> out.append("\\r");
        case '\t' -> out.append("\\t");
        default -> {
          if (c < 0x20) out.append(String.format(Locale.ROOT, "\\u%04x", (int) c));
          else out.append(c);
        }
      }
    }
    return out.toString();
  }

  static String extractJsonString(String json, String key) {
    String needle = "\"" + key + "\"";
    int keyPos = json.indexOf(needle);
    while (keyPos >= 0) {
      int colon = json.indexOf(':', keyPos + needle.length());
      if (colon < 0) return null;
      int quote = colon + 1;
      while (quote < json.length() && Character.isWhitespace(json.charAt(quote))) quote++;
      if (quote < json.length() && json.charAt(quote) == '"') {
        StringBuilder out = new StringBuilder();
        boolean escaped = false;
        for (int i = quote + 1; i < json.length(); i++) {
          char c = json.charAt(i);
          if (escaped) {
            switch (c) {
              case '"', '\\', '/' -> out.append(c);
              case 'b' -> out.append('\b');
              case 'f' -> out.append('\f');
              case 'n' -> out.append('\n');
              case 'r' -> out.append('\r');
              case 't' -> out.append('\t');
              case 'u' -> {
                if (i + 4 >= json.length()) return null;
                String hex = json.substring(i + 1, i + 5);
                try { out.append((char) Integer.parseInt(hex, 16)); }
                catch (NumberFormatException e) { return null; }
                i += 4;
              }
              default -> out.append(c);
            }
            escaped = false;
          } else if (c == '\\') {
            escaped = true;
          } else if (c == '"') {
            return out.toString();
          } else {
            out.append(c);
          }
        }
        return null;
      }
      keyPos = json.indexOf(needle, keyPos + needle.length());
    }
    return null;
  }

  static String trimBody(String body) {
    if (body == null) return "";
    String normalized = body.replaceAll("\\s+", " ").trim();
    return normalized.length() <= 800 ? normalized : normalized.substring(0, 800) + "…";
  }

  static double clamp(double value, double min, double max) {
    return Math.max(min, Math.min(max, value));
  }

  static String formatNumber(double value) {
    if (Math.rint(value) == value && Math.abs(value) < 9e15) return Long.toString((long) value);
    return String.format(Locale.ROOT, "%.12g", value);
  }
}
