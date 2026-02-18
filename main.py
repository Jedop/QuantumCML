import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

class QuantumCML:
    def __init__(self, psi_0, V, dt=1e-4, dx=0.01, device='cpu'):
        """
        psi_0: Initial Complex Wavefunction (Tensor of any shape [N], [N,N], [N,N,N])
        V: Potential Energy (Same shape as psi_0)
        """
        self.device = device
        self.R = psi_0.real.float().to(device)
        self.I = psi_0.imag.float().to(device)
        self.V = V.float().to(device)
        
        self.dt = dt
        self.dx = dx
        
        # Stability constant (Kinetic term coeff)
        # Using 0.5 for standard Schrodinger eq (-1/2 del^2)
        self.K = 0.5 * dt / (dx**2) 
        
        # Auto-detect dimensionality
        self.dims = self.R.dim() 

    def get_laplacian(self, tensor):
        """
        Calculates Laplacian for ANY dimension (1D, 2D, 3D...)
        using finite difference stencil with periodic wrapping (handled by roll).
        """
        lap = torch.zeros_like(tensor)
        
        # Loop through every dimension (0 for x, 1 for y, 2 for z...)
        # and apply the 1D central difference rule to it.
        for d in range(self.dims):
            lap += torch.roll(tensor, shifts=1, dims=d) + \
                   torch.roll(tensor, shifts=-1, dims=d) - \
                   (2 * tensor)
        return lap

    def step(self):
        """
        Update rule using Semi-Implicit Euler (Symplectic-ish)
        """
        # 1. Update Real part using current Imaginary Laplacian
        lap_I = self.get_laplacian(self.I)
        self.R += -self.K * lap_I + (self.V * self.I * self.dt)
        
        # 2. Enforce Hard Walls (Dirichlet BC) - Zero out edges
        self.apply_boundaries(self.R)

        # 3. Update Imaginary part using NEW Real Laplacian
        lap_R = self.get_laplacian(self.R)
        self.I += self.K * lap_R - (self.V * self.R * self.dt)

        # 4. Enforce Hard Walls
        self.apply_boundaries(self.I)

    def apply_boundaries(self, tensor):
        # Zero out the edges for 1D, 2D, 3D
        if self.dims == 1:
            tensor[0] = 0; tensor[-1] = 0
        elif self.dims == 2:
            tensor[0, :] = 0; tensor[-1, :] = 0
            tensor[:, 0] = 0; tensor[:, -1] = 0
        elif self.dims == 3:
            tensor[0, :, :] = 0; tensor[-1, :, :] = 0
            tensor[:, 0, :] = 0; tensor[:, -1, :] = 0
            tensor[:, :, 0] = 0; tensor[:, :, -1] = 0

    def get_prob(self):
        return (self.R**2 + self.I**2).cpu().numpy()

    def run(self, steps, snapshot_interval=50):
        frames = []
        for s in range(steps):
            self.step()
            if s % snapshot_interval == 0:
                frames.append(self.get_prob())
        return frames
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate different Quantum Systems using a CML."
    )
    parser.add_argument(
        "D",
        type=int,
        choices=[1, 2, 3],
        help="The Dimension of the Simulation (1, 2, or 3).",
    )
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Executing {args.D}D Simulation on {device}...")

    if args.D == 1:
        # --- SCENARIO 1: 1D FREE PARTICLE ---
        N = 500
        x = torch.linspace(-10, 10, N)
        dx = (x[1] - x[0]).item()
        
        # Initial Wavepacket: exp(-x^2) * exp(ikx)
        psi = torch.exp(-x**2) * torch.exp(1j * 10 * x)
        
        # Normalization for 1D
        norm = torch.sqrt(torch.sum(torch.abs(psi)**2) * dx)
        psi = (psi / norm).to(torch.complex64)
        
        V = torch.zeros(N)
        
        sim = QuantumCML(psi, V, dt=1e-4, dx=dx, device=device)
        frames = sim.run(steps=25000, snapshot_interval=100)
        
        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(x, frames[0], label="T=0", alpha=0.5)
        plt.plot(x, frames[-1], label="T=Final")
        plt.title("1D Wavepacket Dispersion (Free Particle)")
        plt.xlabel("x"); plt.ylabel("|psi|^2")
        plt.legend(); plt.show()

    elif args.D == 2:
        # --- SCENARIO 2: 2D DOUBLE SLIT INTERFERENCE ---
        N = 200
        coords = torch.linspace(-5, 5, N)
        dx = (coords[1] - coords[0]).item()
        X, Y = torch.meshgrid(coords, coords, indexing='ij')
        
        # Initial Wavepacket at bottom, moving UP
        psi = torch.exp(-((X)**2 + (Y + 3)**2)) * torch.exp(1j * 15 * Y)
        
        # Normalization for 2D
        norm = torch.sqrt(torch.sum(torch.abs(psi)**2) * dx**2)
        psi = (psi / norm).to(torch.complex64)
        
        # Double Slit Potential
        V = torch.zeros((N, N))
        V[:, N//2 : N//2 + 2] = 1000.0 # Wall
        V[N//2 - 10 : N//2 - 5, N//2 : N//2 + 2] = 0.0   # Slit 1
        V[N//2 + 5  : N//2 + 10, N//2 : N//2 + 2] = 0.0  # Slit 2
                
        sim = QuantumCML(psi, V, dt=5e-5, dx=dx, device=device)
        frames = sim.run(steps=20000, snapshot_interval=200)

        V_np = V.numpy()
        wall_mask = (V_np > 100).astype(float)
        # Plotting Final Frame
        plt.figure(figsize=(8, 8))

        img = plt.imshow(frames[-1],
                        extent=[-5, 5, -5, 5],
                        origin="lower",
                        cmap='magma')

        plt.imshow(wall_mask,
                extent=[-5, 5, -5, 5],
                origin="lower",
                cmap="gray",
                alpha=0.25)

        plt.title("2D Double Slit Interference")
        plt.colorbar(img, label="Probability Density")  
        plt.show()

    elif args.D == 3:
        # --- SCENARIO 3: 3D GAUSSIAN IN A BOX ---
        S = 64 # Size 64^3 is roughly 262k cells
        coords = torch.linspace(-5, 5, S)
        dx = (coords[1] - coords[0]).item()
        X, Y, Z = torch.meshgrid(coords, coords, coords, indexing='ij')
        
        # Centered Gaussian
        psi = torch.exp(-(X**2 + Y**2 + Z**2)) 
        
        # Normalization for 3D
        norm = torch.sqrt(torch.sum(torch.abs(psi)**2) * dx**3)
        psi = (psi / norm).to(torch.complex64)
        
        V = torch.zeros((S, S, S))
        
        sim = QuantumCML(psi, V, dt=1e-4, dx=dx, device=device)
        
        frames = sim.run(steps=50000, snapshot_interval=50)
        
        # 2D Visualization
        data = frames[-1]
        proj = data.sum(axis=2)  

        plt.figure()
        plt.imshow(proj, origin="lower", cmap="inferno")
        plt.colorbar(label="projected density")
        plt.title("Projection along z")
        plt.show()



    
