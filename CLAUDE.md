# Claude Code Configuration

## Environment Setup

### Conda Environment
- **Environment name**: `optimization`
- **Activation command**: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate optimization` - Then you can run the target python file

### Required packages
- numpy
- matplotlib
- scipy
- gurobipy
- networkx
- geopandas (for geographic data processing)

### Project Structure
- `lib/algorithm.py` - Main LBBD algorithm implementation
- `lib/data.py` - Geographic data handling
- `lib/evaluate.py` - Performance evaluation utilities
- `test_lbbd.py` - Comprehensive 4×4 grid test with visualization

### Key Algorithm Features
- Logic-Based Benders Decomposition (LBBD)
- Multi-cut generation for acceleration
- Parallel subproblem solving
- Efficient elementwise ODD constraints using big-M formulation
- Mixed-Gaussian demand and ODD distributions

### Testing Commands
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate optimization
python test_lbbd.py            # Comprehensive 4×4 grid test with visualization
```

### Recent Fixes
- ✅ Fixed infinite loop in iteration 1 (removed epsilon scaling)
- ✅ Fixed cut addition sequencing issue
- ✅ Improved elementwise ODD constraints efficiency (big-M formulation)
- ✅ Performance profiling completed - identified subproblems (55.6%) and master problem (44.4%) as main bottlenecks
- ✅ Cleaned up CQCP methods: removed legacy version (~137 lines), renamed `_CQCP_benders_updated` → `_CQCP_benders`
- ⚠️ Parallel processing thread safety issue with Gurobi (needs testing)

### Known Issues
- Parallel mode may have thread contention with Gurobi solver
- Test with `parallel=False` if bounds don't improve in parallel mode