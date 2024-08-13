import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, kurtosis, skew, shapiro
import seaborn as sns

# Função para analisar os dados e determinar a natureza predominante
def analyze_data(data):
    kurt = kurtosis(data, fisher=False)
    skewness = skew(data)
    
    if kurt > 3 and skewness > 0:
        return "Leptocúrtica com Concentração em Sigmas Positivos"
    elif kurt > 3 and skewness < 0:
        return "Leptocúrtica com Concentração em Sigmas Negativos"
    elif kurt > 3:
        return "Leptocúrtica Neutra"
    elif kurt < 3 and skewness > 0:
        return "Platocúrtica com Concentração em Sigmas Positivos"
    elif kurt < 3 and skewness < 0:
        return "Platocúrtica com Concentração em Sigmas Negativos"
    elif kurt < 3:
        return "Platocúrtica Neutra"

# Função para identificar outliers usando o método do intervalo interquartil (IQR)
def detect_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers, lower_bound, upper_bound

# Função para calcular e plotar a distribuição normal e a curtose para cada conjunto de dados
def plot_distribution(data, title, outliers, lower_bound, upper_bound):
    mean = np.mean(data)
    std_dev = np.std(data)
    x = np.linspace(min(data), max(data), 1000)
    y = norm.pdf(x, mean, std_dev)
    
    plt.figure(figsize=(12, 8))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g', label='Dados')
    plt.plot(x, y, label='Distribuição Normal', color='darkred')
    
    sns.kdeplot(data, color='blue', label='Estimativa de Densidade de Kernel', linestyle='--')

    for i in range(1, 4):
        plt.axvline(mean - i*std_dev, color='black', linestyle='dashed', linewidth=1)
        plt.axvline(mean + i*std_dev, color='black', linestyle='dashed', linewidth=1)
    
    plt.axhline(0, color='black', linestyle='dotted', linewidth=1)
    plt.axvline(lower_bound, color='orange', linestyle='dotted', linewidth=1, label='Limite Inferior (Outliers)')
    plt.axvline(upper_bound, color='orange', linestyle='dotted', linewidth=1, label='Limite Superior (Outliers)')
    
    plt.scatter(outliers, [0] * len(outliers), color='red', label='Outliers', zorder=5)
    
    plt.text(mean, max(y)*0.5, r'$\mu$', horizontalalignment='center', fontsize=12)
    plt.text(mean - std_dev, max(y)*0.2, r'$\mu - 1\sigma$', horizontalalignment='center', fontsize=12)
    plt.text(mean + std_dev, max(y)*0.2, r'$\mu + 1\sigma$', horizontalalignment='center', fontsize=12)
    plt.text(mean - 2*std_dev, max(y)*0.05, r'$\mu - 2\sigma$', horizontalalignment='center', fontsize=12)
    plt.text(mean + 2*std_dev, max(y)*0.05, r'$\mu + 2\sigma$', horizontalalignment='center', fontsize=12)
    plt.text(mean - 3*std_dev, max(y)*0.01, r'$\mu - 3\sigma$', horizontalalignment='center', fontsize=12)
    plt.text(mean + 3*std_dev, max(y)*0.01, r'$\mu + 3\sigma$', horizontalalignment='center', fontsize=12)
    
    kurt = kurtosis(data, fisher=False)
    skewness = skew(data)
    _, p_value = shapiro(data)
    
    plt.title(f'{title}\nCurtose: {kurt:.2f}, Assimetria: {skewness:.2f}, p-valor do Shapiro-Wilk: {p_value:.4f}')
    plt.xlabel('Valor')
    plt.ylabel('Densidade de Probabilidade')
    plt.legend()
    plt.show()

# Função para gerar dados com base na escolha do usuário
def generate_data(leptocurtic=True, skew_type="neutral"):
    if leptocurtic:
        base_data = np.concatenate([np.random.normal(0, 0.5, 950), np.random.normal(0, 2, 50)])
    else:
        base_data = np.concatenate([np.random.normal(-2, 1, 500), np.random.normal(2, 1, 500)])
    
    if skew_type == "negative":
        skew_data = np.concatenate([np.random.normal(-2, 0.5, 900), np.random.normal(1, 0.5, 100)])
    elif skew_type == "positive":
        skew_data = np.concatenate([np.random.normal(2, 0.5, 900), np.random.normal(-1, 0.5, 100)])
    else:
        skew_data = np.random.normal(0, 1, 1000)
    
    data = np.concatenate([base_data, skew_data])
    return data

# Função para gerar gráficos para todas as combinações possíveis
def generate_all_plots():
    distributions = {
        'Leptocúrtica': True,
        'Platocúrtica': False
    }
    
    skews = {
        'Sigmas Negativos': 'negative',
        'Neutro': 'neutral',
        'Sigmas Positivos': 'positive'
    }
    
    for dist_name, leptocurtic in distributions.items():
        for skew_name, skew_type in skews.items():
            print(f'\nGerando gráficos para {dist_name} com {skew_name}...')
            data = generate_data(leptocurtic=leptocurtic, skew_type=skew_type)
            nature = analyze_data(data)
            outliers, lower_bound, upper_bound = detect_outliers(data)
            plot_distribution(data, f'{dist_name} com {skew_name}', outliers, lower_bound, upper_bound)

# Interação com o usuário para gerar gráficos
print("Você deseja gerar gráficos para todas as combinações possíveis? (s/n)")
user_choice = input("Escolha s ou n: ")

if user_choice.lower() == 's':
    generate_all_plots()
else:
    print("Escolha o tipo de distribuição base:")
    print("1 - Leptocúrtica")
    print("2 - Platocúrtica")
    base_choice = input("Escolha 1 ou 2: ")

    print("\nEscolha a concentração de sigmas:")
    print("1 - Sigmas Negativos")
    print("2 - Neutro")
    print("3 - Sigmas Positivos")
    skew_choice = input("Escolha 1, 2 ou 3: ")

    leptocurtic = base_choice == "1"
    skew_type = "neutral"
    if skew_choice == "1":
        skew_type = "negative"
    elif skew_choice == "3":
        skew_type = "positive"

    data = generate_data(leptocurtic=leptocurtic, skew_type=skew_type)

    nature = analyze_data(data)
    print(f'\nNatureza predominante dos dados: {nature}')

    outliers, lower_bound, upper_bound = detect_outliers(data)
    plot_distribution(data, nature, outliers, lower_bound, upper_bound)
