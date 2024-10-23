from hierarchicalforecast.methods import MinTrace, BottomUp, OptimalCombination, ERM

class ReconcilerFactory:
    """Factory class to create reconciliation methods based on user selection."""
    
    def create_reconciler(self, method: str):
        """Creates and returns a reconciliation method based on the provided name."""
        if method == 'MinTraceOLS':
            return MinTrace(method='ols')
        elif method == 'MinTraceShrink':
            return MinTrace(method='mint_shrink')
        elif method == 'BottomUp':
            return BottomUp()
        elif method == 'OptimalCombinationWLSStruct':
            return OptimalCombination(method='wls_struct')
        elif method == 'ERMClosed':
            return ERM(method='closed')
        elif method == 'ERMReg':
            return ERM(method='reg')
        elif method == 'ERMRegBU':
            return ERM(method='reg_bu')
        elif method == 'MinTraceWLSStruct':
            return MinTrace(method='wls_struct')
        elif method == 'MinTraceWLSVar':
            return MinTrace(method='wls_var')
        else:
            raise ValueError(f"Unknown reconciliation method: {method}")
