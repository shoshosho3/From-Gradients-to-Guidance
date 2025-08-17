# Sanity Check: Uncertainty vs Random Active Learning

This directory contains scripts to verify that the uncertainty-based active learning strategy (which should be SOTA) consistently beats random sampling.

## 🎯 **Purpose**
The sanity check ensures that our uncertainty-based active learning implementation is working correctly by demonstrating that it consistently outperforms random sampling across multiple experimental runs.

## 📁 **Files**

### 1. **`src/uncertainty_strategy.py`** - Full Sanity Check
- **Purpose**: Complete comparison between uncertainty and random strategies
- **Configuration**: 20 rounds, 20 epochs per round, 5 seeds
- **Expected Runtime**: ~2-4 hours (depending on hardware)
- **Output**: Detailed comparison, plots, and saved results

### 2. **`src/quick_sanity_test.py`** - Quick Test
- **Purpose**: Fast verification with reduced parameters
- **Configuration**: 5 rounds, 5 epochs per round, 2 seeds, full CIFAR-10
- **Expected Runtime**: ~15-30 minutes
- **Output**: Quick comparison and basic plot

### 3. **`src/tiny_sanity_test.py`** - Tiny Test
- **Purpose**: Ultra-fast verification with minimal parameters
- **Configuration**: 3 rounds, 2 epochs per round, 2 seeds, 1,000 samples
- **Expected Runtime**: ~2-5 minutes
- **Output**: Quick comparison and basic plot

## 🚀 **How to Run**

### Tiny Test (Recommended First)
```bash
cd src
python tiny_sanity_test.py
```

### Quick Test
```bash
cd src
python quick_sanity_test.py
```

### Full Sanity Check
```bash
cd src
python uncertainty_strategy.py
```

## 📊 **What to Expect**

### **Success Criteria**
- ✅ **PASS**: Uncertainty strategy wins >70% of rounds
- ⚠️ **PARTIAL**: Uncertainty strategy wins >50% of rounds  
- ❌ **FAIL**: Random strategy wins too many rounds

### **Expected Results**
- Uncertainty strategy should consistently achieve higher accuracy
- Performance gap should increase with more labeled samples
- Results should be consistent across different random seeds

## 🔍 **How It Works**

### **Uncertainty Strategy**
1. **Entropy-based**: Measures prediction entropy (higher = more uncertain)
2. **Margin-based**: Measures confidence gap between top-2 predictions
3. **Combined Score**: Entropy + (1 - margin) for robust uncertainty estimation
4. **Selection**: Always picks the most uncertain samples

### **Random Strategy**
1. **Baseline**: Randomly samples from unlabeled pool
2. **Control**: Ensures fair comparison with uncertainty approach

### **Evaluation**
- Both strategies use identical training procedures
- Same model architecture (ResNet18)
- Same hyperparameters and data splits
- Only difference is sample selection strategy

## 📈 **Output Files**

### **Generated Files**
- `sanity_check_uncertainty_vs_random.png` - Comparison plot
- `sanity_check_results.npy` - Detailed results (numpy format)
- Console output with round-by-round comparison

### **Console Output Example**
```
=== RESULTS COMPARISON ===
Labeled Samples | Uncertainty | Random   | Difference
--------------------------------------------------
           100 |     0.2345 |  0.1987 | +0.0358 ✓ UNC wins
           150 |     0.3456 |  0.2876 | +0.0580 ✓ UNC wins
           200 |     0.4123 |  0.3567 | +0.0556 ✓ UNC wins
           250 |     0.4567 |  0.3987 | +0.0580 ✓ UNC wins
           300 |     0.4987 |  0.4456 | +0.0531 ✓ UNC wins

=== SANITY CHECK VERDICT ===
Uncertainty strategy wins: 5/5 rounds
✅ SANITY CHECK PASSED: Uncertainty strategy consistently beats random!
```

## 🐛 **Troubleshooting**

### **If Random Wins Too Often**
1. Check uncertainty computation implementation
2. Verify entropy and margin calculations
3. Ensure proper model evaluation
4. Check for data leakage or bugs

### **If Results Are Inconsistent**
1. Verify random seed setting
2. Check for non-deterministic operations
3. Ensure proper data splitting
4. Verify model initialization

## 🔬 **Technical Details**

### **Uncertainty Measures**
- **Entropy**: `-Σ(p_i * log(p_i))` where p_i are softmax probabilities
- **Margin**: `p_max1 - p_max2` where p_max1, p_max2 are top-2 probabilities
- **Combined**: `entropy + (1 - margin)` for balanced uncertainty estimation

### **Model Architecture**
- **Backbone**: Pre-trained ResNet18
- **Head**: Linear classification layer
- **Training**: SGD with momentum, CrossEntropyLoss
- **Evaluation**: Accuracy on validation set

### **Data Handling**
- **Dataset**: CIFAR-10 (50,000 train, 10,000 test)
- **Initial**: 100 labeled samples
- **Query Size**: 50 samples per round
- **Transforms**: Resize to 224x224, ToTensor

## 📚 **References**

This sanity check is based on established active learning literature where uncertainty-based sampling typically outperforms random sampling:
- Entropy-based uncertainty sampling
- Margin-based uncertainty sampling  
- Expected model change approaches
- Information-theoretic sample selection

## 🎉 **Success Indicators**

When the sanity check passes, you can be confident that:
1. Your uncertainty implementation is working correctly
2. The active learning pipeline is functioning properly
3. Your MINDS algorithm has a solid baseline to compare against
4. The experimental setup is ready for more advanced comparisons
