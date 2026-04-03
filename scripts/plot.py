import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. قراءة البيانات
data = pd.read_csv('benchmark_results.csv')

# 2. الحصول على اسم المعالج (CPU Model)
# لو مش موجود هيحط "Unknown CPU"
cpu_model = os.getenv('CPU_MODEL', 'Standard AVX2 CPU')

# 3. إعداد الرسم البياني
plt.figure(figsize=(14, 7))
plt.plot(data['Size'], data['DNNL_GFLOPS'], label='Intel oneDNN (DNNL)', 
         marker='o', linestyle='--', linewidth=2, color='#1f77b4')
plt.plot(data['Size'], data['ME_GFLOPS'], label='My Optimized Kernel (ME)', 
         marker='s', linestyle='-', linewidth=2, color='#ff7f0e')

# 4. تجميل الرسم البياني
plt.title(f'Performance Comparison: My GEMM vs Intel oneDNN\nHardware: {cpu_model}', fontsize=14, fontweight='bold')
plt.xlabel('Matrix Size (M=N=K)', fontsize=12)
plt.ylabel('Performance (GFLOPS)', fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend(loc='upper left', frameon=True, shadow=True)

# إضافة خط السقف (Optional لو عايز تبين الـ Peak)
# plt.axhline(y=350, color='r', linestyle=':', label='Approx. HW Peak')

# 5. حفظ الرسم البياني وعرضه
plt.savefig('performance_plot.png', dpi=300, bbox_inches='tight')
print(f"✅ Plot saved as performance_plot.png for {cpu_model}")