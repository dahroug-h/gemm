import pandas as pd
import matplotlib.pyplot as plt
import os

try:
    # 1. قراءة البيانات من ملف الـ CSV المحدث
    # نتوقع وجود أعمدة: Size, DNNL_GFLOPS, ME_GFLOPS, ROBUST_GFLOPS
    data = pd.read_csv('benchmark_results.csv')

    # 2. الحصول على اسم المعالج من البيئة المحيطة
    cpu_model = os.getenv('CPU_MODEL', 'AVX2 Optimized CPU')

    # 3. إعداد الرسم البياني باحترافية للبلوج
    plt.figure(figsize=(15, 8))
    
    # رسم خط إنتل (المرجع الذي يفشل في الرينج الكبير)
    plt.plot(data['Size'], data['DNNL_GFLOPS'], label='Intel oneDNN (Standard)', 
             marker='o', linestyle='--', linewidth=2, color='#1f77b4', alpha=0.6)
    
    # رسم كودك القياسي (الأداء العالي)
    plt.plot(data['Size'], data['ME_GFLOPS'], label='ME Standard Optimized', 
             marker='s', linestyle='-', linewidth=2.5, color='#ff7f0e')
    
    # رسم كودك المتين (البطل الذي يدعم الرينج الكامل [-128, 127])
    if 'ROBUST_GFLOPS' in data.columns:
        plt.plot(data['Size'], data['ROBUST_GFLOPS'], label='ME Robust (Safe Full Range)', 
                 marker='D', linestyle='-', linewidth=3, color='#2ca02c')

    # 4. إضافة منطقة "فشل الدقة" لإنتل
    # بما أن إنتل تبدأ في الخطأ عند تجاوز +-28، نوضح ذلك بصرياً
    plt.axhspan(0, data['DNNL_GFLOPS'].max() + 50, alpha=0.05, color='red', 
                label='oneDNN Numerical Instability Zone (> ±28)')

    # 5. تجميل الرسم البياني
    plt.title(f'Breaking the AVX2 Barrier: Performance & Robustness\nHardware: {cpu_model}', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Matrix Size (M=N=K)', fontsize=12, fontweight='bold')
    plt.ylabel('Performance (GFLOPS)', fontsize=12, fontweight='bold')
    
    # إضافة شبكة بيانية وتنسيق الأسطورة
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(loc='upper left', frameon=True, shadow=True, fontsize=10)

    # إضافة ملحوظة تقنية على الرسمة
    plt.annotate('Robust Kernel: 100% Precision for [-128, 127]', 
                 xy=(data['Size'].iloc[-1], data['ROBUST_GFLOPS'].iloc[-1] if 'ROBUST_GFLOPS' in data.columns else 200),
                 xytext=(-250, 30), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='green'),
                 fontsize=10, color='green', fontweight='bold')

    # 6. حفظ الرسم البياني بجودة عالية للطباعة
    plt.savefig('performance_plot.png', dpi=300, bbox_inches='tight')
    print(f"✅ Full professional plot saved as performance_plot.png for {cpu_model}")

except Exception as e:
    print(f"❌ Error generating full plot: {e}")