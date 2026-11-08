import SwiftUI

@main
struct NeuralArenaBitwiseApp: App {
    var body: some Scene {
        WindowGroup { DeviceValidationView() }
    }
}

struct DeviceValidationView: View {
    @State private var status = "Device verification not yet executed"
    @State private var evidenceURL: URL?
    @State private var running = false
    @State private var log = "Target: iPhone 15 Pro Max (A17 Pro, Apple GPU family 9).\n27 golden-answer outputs across 9 3D cases.\nDual GPU kernels: direct + tiled-halo.\nRun on the physical device to claim a device PASS."

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    Text("NEURAL ARENA ΩΞ").font(.title).bold()
                    Text("Bitwise GPU Conformance • Q12 • 3D").font(.headline)
                    Text(status).foregroundStyle(status.hasPrefix("PASS") ? .green : .primary)
                        .font(.headline)
                    Button(running ? "Running 3D validation…" : "Run A17 Pro GPU Bitwise Validation") {
                        run()
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(running)
                    if let url = evidenceURL {
                        ShareLink(item: url) { Label("Export device evidence JSON", systemImage: "square.and.arrow.up") }
                            .buttonStyle(.bordered)
                    }
                    Text(log).font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }.padding(20)
            }.navigationTitle("Device Conformance")
        }
    }
    private func run() {
        running=true
        status="Executing CPU and GPU comparison…"
        evidenceURL=nil
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let harness=try NeuralArenaDeviceHarness()
                let result=harness.run()
                let encoder=JSONEncoder()
                encoder.outputFormatting=[.prettyPrinted,.sortedKeys]
                let json=try encoder.encode(result)
                let url=FileManager.default.temporaryDirectory.appendingPathComponent("NeuralArena_A17Pro_Bitwise_Evidence.json")
                try json.write(to:url,options:.atomic)
                let summary="GPU: \(result.gpuName) | Model \(result.hardwareModel)\nApple family 9: \(result.supportsAppleGPUFamily9) | Simulator: \(result.isSimulator)\n\nMatched: \(result.casesPassed)/\(result.casesTotal) CPU/direct GPU/tiled GPU/golden steps.\n\n" + (result.exceptions.isEmpty ? "All 27 comparisons passed.\n" : result.exceptions.joined(separator:"\n")+"\n") + "\nHardware verification flag: \(result.deviceBitwiseValidated)"
                DispatchQueue.main.async {
                    self.log=summary
                    self.status=result.deviceBitwiseValidated ? "PASS — A17 Pro hardware verified" : (result.allComparisonsPassed ? "PASS — arithmetic; target not A17 Pro" : "FAIL — see mismatches")
                    self.evidenceURL=url;self.running=false
                }
            } catch {
                DispatchQueue.main.async {
                    self.status="FAIL — device verification could not run"
                    self.log="\(error)\n\nEnsure Metal shaders compile into the app's default library."
                    self.running=false
                }
            }
        }
    }
}
