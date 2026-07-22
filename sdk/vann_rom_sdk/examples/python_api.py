import numpy as np

from vann_rom_sdk import Assembler, TinyAutoencoder, VANNVirtualMachine

source = open("examples/demo.vann", encoding="utf-8").read()
program = Assembler().assemble(source)

model = TinyAutoencoder(input_dim=12, latent_dim=4)
vm = VANNVirtualMachine(model, output_sink=print)
vm.load_program(program.instructions)
vm.set_input(np.array([[0.1, 0.2, 0.9, 0.8, 0.15, 0.05, 0.7, 0.65, 0.4, 0.3, 0.95, 0.55]]))
result = vm.run()

print(result.metrics)
print(result.policy)
