import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr
import os

# Change working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"Changed working directory to: {os.getcwd()}")

data = pd.read_csv('deidentified_data.csv')

# Create a PDF file to save all figures
with PdfPages('bent_analysis_figures.pdf') as pdf:
    
    # Figure 1: Overall scatter plots with swapped axes
    plt.rcParams["figure.figsize"] = (20,3)
    fig,axs = plt.subplots(1,6)
    devs = data.columns[1:7]
    correlations_overall = {}
    
    for n in range(6):
        dev = devs[n]
        i = (data['ECG']==data['ECG']) & (data[dev]==data[dev])
        x = np.array(data.loc[i][dev])  # Device on x-axis
        y = np.array(data.loc[i]['ECG'])  # ECG on y-axis
        
        # Calculate Pearson correlation
        r, p_value = pearsonr(x, y)
        correlations_overall[dev] = {'r': r, 'p': p_value, 'n': len(x)}
        
        # Format p-value for readability
        p_text = "p < .001" if p_value < 0.001 else f"p = {p_value:.3f}"
        
        ax = axs.flatten()[n]
        ax.scatter(x,y,s=0.0001)
        ax.set_title(f'{dev} (N = {len(x)})\nr = {r:.3f}, {p_text}')
        ax.set_xlabel('Device [bpm]')
        if n==0:
            ax.set_ylabel('ECG [bpm]')
        ax.set_xlim((40,180))
        ax.set_ylim((40,180))
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.show()
    
    # Print overall correlations
    print("\nOverall Correlations:")
    for dev, stats in correlations_overall.items():
        p_text = "p < .001" if stats['p'] < 0.001 else f"p = {stats['p']:.3f}"
        print(f"{dev}: r = {stats['r']:.4f}, {p_text}, N = {stats['n']}")

    # Figure 2: Lighter skin tone scatter plots with swapped axes
    plt.rcParams["figure.figsize"] = (20,3)
    fig,axs = plt.subplots(1,6)
    devs = data.columns[1:7]
    correlations_lighter = {}
    
    for n in range(6):
        dev = devs[n]
        i = (data['ECG']==data['ECG']) & (data[dev]==data[dev]) & (data['Skin Tone']<4)
        x = np.array(data.loc[i][dev])  # Device on x-axis
        y = np.array(data.loc[i]['ECG'])  # ECG on y-axis
        
        # Calculate Pearson correlation
        r, p_value = pearsonr(x, y)
        correlations_lighter[dev] = {'r': r, 'p': p_value, 'n': len(x)}
        
        # Format p-value for readability
        p_text = "p < .001" if p_value < 0.001 else f"p = {p_value:.3f}"
        
        ax = axs.flatten()[n]
        ax.scatter(x,y,s=0.0001)
        ax.set_title(f'{dev} (Lighter N={len(x)})\nr = {r:.3f}, {p_text}')
        ax.set_xlabel('Device [bpm]')
        if n==0:
            ax.set_ylabel('ECG [bpm]')
        ax.set_xlim((40,180))
        ax.set_ylim((40,180))
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.show()
    
    # Print lighter skin tone correlations
    print("\nLighter Skin Tone Correlations:")
    for dev, stats in correlations_lighter.items():
        p_text = "p < .001" if stats['p'] < 0.001 else f"p = {stats['p']:.3f}"
        print(f"{dev}: r = {stats['r']:.4f}, {p_text}, N = {stats['n']}")

    # Figure 3: Darker skin tone scatter plots with swapped axes
    plt.rcParams["figure.figsize"] = (20,3)
    fig,axs = plt.subplots(1,6)
    devs = data.columns[1:7]
    correlations_darker = {}
    
    for n in range(6):
        dev = devs[n]
        i = (data['ECG']==data['ECG']) & (data[dev]==data[dev]) & (data['Skin Tone']>3)
        x = np.array(data.loc[i][dev])  # Device on x-axis
        y = np.array(data.loc[i]['ECG'])  # ECG on y-axis
        
        # Calculate Pearson correlation
        r, p_value = pearsonr(x, y)
        correlations_darker[dev] = {'r': r, 'p': p_value, 'n': len(x)}
        
        # Format p-value for readability
        p_text = "p < .001" if p_value < 0.001 else f"p = {p_value:.3f}"
        
        ax = axs.flatten()[n]
        ax.scatter(x,y,s=0.0001)
        ax.set_title(f'{dev} (Darker N={len(x)})\nr = {r:.3f}, {p_text}')
        ax.set_xlabel('Device [bpm]')
        if n==0:
            ax.set_ylabel('ECG [bpm]')
        ax.set_xlim((40,180))
        ax.set_ylim((40,180))
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.show()
    
    # Print darker skin tone correlations
    print("\nDarker Skin Tone Correlations:")
    for dev, stats in correlations_darker.items():
        p_text = "p < .001" if stats['p'] < 0.001 else f"p = {stats['p']:.3f}"
        print(f"{dev}: r = {stats['r']:.4f}, {p_text}, N = {stats['n']}")

    # Figure 4: Histograms with fixed y-axis limit
    plt.rcParams["figure.figsize"] = (20,3)
    fig,axs = plt.subplots(1,6)
    devs = data.columns[1:7]
    for n in range(6):
        dev = devs[n]
        i = (data['ECG']==data['ECG']) & (data[dev]==data[dev])
        x = np.array(data.loc[i]['ECG'])
        ax = axs.flatten()[n]
        ax.hist(x,bins=np.arange(40,180,10))
        ax.set_title(f'{dev} (N = {len(x)})')
        ax.set_xlabel('ECG [bpm]')
        if n==0:
            ax.set_ylabel('N')
        ax.set_ylim((0,35000))
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.show()

    # Figure 5: Histograms with auto y-axis scaling
    plt.rcParams["figure.figsize"] = (20,3)
    fig,axs = plt.subplots(1,6)
    devs = data.columns[1:7]
    for n in range(6):
        dev = devs[n]
        i = (data['ECG']==data['ECG']) & (data[dev]==data[dev])
        x = np.array(data.loc[i]['ECG'])
        ax = axs.flatten()[n]
        ax.hist(x,bins=np.arange(40,180,10))
        ax.set_title(f'{dev} (N = {len(x)})')
        ax.set_xlabel('ECG [bpm]')
        if n==0:
            ax.set_ylabel('N')
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.show()

    # Figure 6: Percentage histograms
    plt.rcParams["figure.figsize"] = (20,3)
    fig,axs = plt.subplots(1,6)
    devs = data.columns[1:7]
    for n in range(6):
        dev = devs[n]
        i = (data['ECG']==data['ECG']) & (data[dev]==data[dev])
        x = np.array(data.loc[i]['ECG'])
        ax = axs.flatten()[n]
        ax.hist(x,bins=np.arange(40,180,5),weights=100*np.ones(len(x))/len(x))
        ax.set_title(f'{dev} (N = {len(x)})')
        ax.set_xlabel('ECG [bpm]')
        if n==0:
            ax.set_ylabel('%')
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.show()

print("All figures saved to 'bent_analysis_figures.pdf'")
print(f"Number of unique IDs: {len(set(data['ID']))}")

# Summary table of correlations
print("\n" + "="*80)
print("CORRELATION SUMMARY TABLE")
print("="*80)
print(f"{'Device':<15} {'Overall r':<12} {'Lighter r':<12} {'Darker r':<12} {'Difference':<12}")
print("-"*80)

for dev in data.columns[1:7]:
    overall_r = correlations_overall.get(dev, {}).get('r', 0)
    lighter_r = correlations_lighter.get(dev, {}).get('r', 0)
    darker_r = correlations_darker.get(dev, {}).get('r', 0)
    diff = lighter_r - darker_r
    print(f"{dev:<15} {overall_r:<12.4f} {lighter_r:<12.4f} {darker_r:<12.4f} {diff:<12.4f}")

print("="*80)