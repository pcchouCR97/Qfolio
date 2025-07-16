from qiskit_optimization import QuadraticProgram
from qiskit_optimization.applications import OptimizationApplication
from qiskit_optimization.translators import from_docplex_mp
from docplex.mp.model import Model as AdvModel
from utilities import estimate_num_bits_basek
import numpy as np

class PortfolioOptimizationHamiltonian(OptimizationApplication):
    def __init__(
            self,
            expected_returns: np.ndarray,
            covariances: np.ndarray,
            q: float,
            budget: float,  # Total budget in monetary value
            lambda1: float,
            lambda2: float,
            prices: np.ndarray, # Price per asset
            max_binary_bits: int,  # Number of binary bits per asset (adjustable)      
            expected_returns_rate: float # Should be a reasonable number
    ) -> None:
        self._expected_returns = expected_returns
        self._covariances = covariances
        self._budget = budget
        self.q = q
        self._lambda1 = lambda1
        self._lambda2 = lambda2
        self._prices = prices  
        self._max_binary_bits = max_binary_bits  # Define number of bits for binary encoding
        self.expected_returns_rate = expected_returns_rate

    def to_quadratic_program(self) -> QuadraticProgram:
        num_assets = len(self._expected_returns)
        mdl = AdvModel(name=f"Portfolio Optimization ")

        # Binary decision variables for encoding the number of shares
        x = {
            (i, j): mdl.binary_var(name=f"x_{i}_{j}") 
            for i in range(num_assets) 
            for j in range(self._max_binary_bits)
        }
        
        # Convert binary representation to integer share count
        s = {
            i: mdl.sum(2**j * x[i, j] for j in range(self._max_binary_bits))
            for i in range(num_assets)
        }

        # Quadratic risk term using covariance matrix
        h_1 = mdl.sum(
            self._covariances[i, j] * self._prices[i] * s[i] * self._prices[j] * s[j] 
            for i in range(num_assets) 
            for j in range(num_assets)
        )

        # Budget constraint: Ensure total cost is within available capital
        h_2 = (mdl.sum(s[i] * self._prices[i] for i in range(num_assets)) - self._budget) ** 2
        
        # Expected returns penalty (maximize returns) 
        h_3 = (self.expected_returns_rate * self._budget - mdl.sum(self._expected_returns[i] * s[i] for i in range(num_assets)))**2
        
        # Final Hamiltonian function
        H = self.q * h_1 + self._lambda1 * h_2 + self._lambda2 * h_3  

        # Minimize the function
        mdl.minimize(H)

        # Convert to Qiskit's Quadratic Program
        op = from_docplex_mp(mdl)
        return op
    
    def interpret(self, result) -> list:
        """Extracts the number of shares from the binary solution."""
        num_assets = len(self._expected_returns)
        num_bits = self._max_binary_bits
        shares = {}
        
        for i in range(num_assets):
            # Decode binary solution to integer share count
            share_value = sum(2**j * int(result.x[i * num_bits + j]) for j in range(num_bits))
            if share_value > 0:
                shares[i] = share_value
        
        return shares
    
    def to_quadratic_program_k(self,
                               k) -> QuadraticProgram:
        num_assets = len(self._expected_returns)
        mdl = AdvModel(name=f"Portfolio Optimization ")

        # Binary decision variables for encoding the number of shares
        x = {
            (i, j): mdl.binary_var(name=f"x_{i}_{j}") 
            for i in range(num_assets) 
            for j in range(self._max_binary_bits)
        }
        
        # Convert binary representation to integer share count
        s = {
            i: mdl.sum(k**j * x[i, j] for j in range(self._max_binary_bits))
            for i in range(num_assets)
        }

        # Quadratic risk term using covariance matrix
        h_1 = mdl.sum(
            self._covariances[i, j] * self._prices[i] * s[i] * self._prices[j] * s[j] 
            for i in range(num_assets) 
            for j in range(num_assets)
        )

        # Budget constraint: Ensure total cost is within available capital
        h_2 = (mdl.sum(s[i] * self._prices[i] for i in range(num_assets)) - self._budget) ** 2
        
        # Expected returns penalty (maximize returns) 
        h_3 = (self.expected_returns_rate * self._budget - mdl.sum(self._expected_returns[i] * s[i] for i in range(num_assets)))**2
        
        # Final Hamiltonian function
        H = self.q * h_1 + self._lambda1 * h_2 + self._lambda2 * h_3  

        # Minimize the function
        mdl.minimize(H)

        # Convert to Qiskit's Quadratic Program
        op = from_docplex_mp(mdl)
        return op
    
    def interpret_k(self, result, k) -> list:
        """Extracts the number of shares from the binary solution."""
        num_assets = len(self._expected_returns)
        num_bits = self._max_binary_bits
        shares = {}
        
        for i in range(num_assets):
            # Decode binary solution to integer share count
            share_value = sum(k**j * int(result.x[i * num_bits + j]) for j in range(num_bits))
            if share_value > 0:
                shares[i] = share_value
        
        return shares


# Test 
if __name__ == "__main__":
    # Import dependencies
    import numpy as np

    # Example data for testing
    expected_returns = np.array([0.1, 0.2, 0.15])
    covariances = np.array([[0.1, 0.02, 0.01], 
                             [0.02, 0.08, 0.03], 
                             [0.01, 0.03, 0.09]])
    prices = np.array([50, 300, 500])
    budget = 10000
    lambda1 = 100
    lambda2 = 100
    
    base_k = 2
    num_bits = estimate_num_bits_basek(budget, prices, k = base_k)
    max_binary_bits = num_bits
    err = 0.5
    # Instantiate class with test values
    optimizer = PortfolioOptimizationHamiltonian(
        expected_returns=expected_returns,
        covariances=covariances,
        q=0.5,
        budget=budget,
        lambda1=lambda1,
        lambda2 = lambda2,
        prices=prices,
        max_binary_bits= max_binary_bits,      
        expected_returns_rate = err 
    )

    # Test if `to_quadratic_program()` runs correctly
    qp = optimizer.to_quadratic_program_k(k= base_k)
    
    # Print result to confirm it's working
    print("Test successful! Quadratic Program generated:")
    print(num_bits)
    print(qp)

