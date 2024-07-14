import itertools
import nltk
import matplotlib.pyplot as plt
import numpy as np
import os
import webbrowser
from datetime import datetime
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import RSLPStemmer
from nltk.probability import FreqDist
from base import raiva, tristeza, surpresa, medo, desgosto, alegria

classificador = None
palavrasunicas = None

def count_paragraphs(text: str) -> list:
    """Conta parágrafos no texto, retornando apenas os parágrafos não vazios.
    
    Args:
        text (str): Texto a ser analisado.

    Returns:
        list: Lista de parágrafos não vazios.
    """
    paragraphs = text.split('\n\n')
    return [p for p in paragraphs if p.strip()]

def count_sentences(text: str) -> list:
    """Conta frases no texto utilizando o tokenizador de sentenças do NLTK.

    Args:
        text (str): Texto a ser analisado.

    Returns:
        list: Lista de frases tokenizadas.
    """
    return sent_tokenize(text, language='portuguese')

class SentimentAnalyzer:
    """
    Classe para análise de sentimentos em textos utilizando processamento de linguagem natural.
    """
    def __init__(self):
        """Inicializa o analisador de sentimentos, baixando recursos necessários do NLTK."""
        nltk.download('stopwords')
        nltk.download('punkt')
        self.stopwordsnltk = nltk.corpus.stopwords.words('portuguese')
        self.emotions_funcs = {
            'raiva': raiva,
            'tristeza': tristeza,
            'surpresa': surpresa,
            'medo': medo,
            'desgosto': desgosto,
            'alegria': alegria
        }
        os.makedirs('generated_html/generated_images', exist_ok=True)

    def process_document(self, filepath: str) -> tuple:
        """Processa um documento para separar em parágrafos e frases.

        Args:
            filepath (str): Caminho do arquivo a ser processado.

        Returns:
            tuple: Tupla contendo uma lista de parágrafos e uma lista de frases.
        """
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
        paragraphs = count_paragraphs(text)
        sentences = count_sentences(text)
        return paragraphs, sentences

    def analyze_paragraphs(self, paragraphs: list) -> tuple:
        """Analisa cada parágrafo para obter pontuações de sentimentos.

        Args:
            paragraphs (list): Lista de parágrafos para análise.

        Returns:
            tuple: Tupla contendo uma lista de pontuações de sentimentos e índices do final dos parágrafos.
        """
        scores_list = []
        paragraph_end_indices = []
        sentence_count = 0
        for paragraph in paragraphs:
            sents = count_sentences(paragraph)
            for sentence in sents:
                scores = self.classify_emotion(sentence)
                scores_list.append(scores)
            sentence_count += len(sents)
            paragraph_end_indices.append(sentence_count)
        return scores_list, paragraph_end_indices

    def aplicastemmer(self, texto: list) -> list:
        """Aplica um algoritmo de stemming às palavras do texto.

        Args:
            texto (list): Lista de tuplas contendo palavras e suas respectivas emoções.

        Returns:
            list: Lista de tuplas com palavras após aplicação de stemming e suas emoções.
        """
        stemmer = RSLPStemmer()
        frasesstemming = []
        for (palavras, emocao) in texto:
            comstemming = [str(stemmer.stem(p)) for p in word_tokenize(palavras) if p not in self.stopwordsnltk]
            frasesstemming.append((comstemming, emocao))
        return frasesstemming

    def buscapalavras(self, frases: list) -> list:
        """Busca todas as palavras fornecidas nas frases.

        Args:
            frases (list): Lista de frases para extração de palavras.

        Returns:
            list: Lista de palavras extraídas das frases.
        """
        all_words = []
        for (words, _) in frases:
            all_words.extend(words)
        return all_words

    def buscafrequencia(self, palavras: list) -> FreqDist:
        """Calcula a frequência das palavras fornecidas.

        Args:
            palavras (list): Lista de palavras.

        Returns:
            FreqDist: Distribuição de frequência das palavras.
        """
        freq = FreqDist(palavras)
        return freq

    def buscapalavrasunicas(self, frequencia: FreqDist) -> list:
        """Busca palavras únicas a partir da frequência de palavras.

        Args:
            frequencia (FreqDist): Distribuição de frequência das palavras.

        Returns:
            list: Lista de palavras únicas.
        """
        return list(frequencia.keys())

    def extratorpalavras(self, documento: list) -> dict:
        """Extrai palavras do documento comparando com palavras únicas.

        Args:
            documento (list): Lista de palavras do documento.

        Returns:
            dict: Dicionário de características das palavras.
        """
        doc = set(documento)
        features = {word: (word in doc) for word in palavrasunicas}
        return features

    def classify_emotion(self, sentence: str) -> np.array:
        """Classifica uma frase para cada emoção usando um Classificador Bayesiano Ingênuo.

        Args:
            sentence (str): Frase a ser classificada.

        Returns:
            np.array: Array de probabilidades para cada emoção.
        """
        global classificador, palavrasunicas
        if not classificador:
            training_base = sum((self.emotions_funcs[emotion]() for emotion in self.emotions_funcs), [])
            frasesstemming = self.aplicastemmer(training_base)
            palavras = self.buscapalavras(frasesstemming)
            frequencia = self.buscafrequencia(palavras)
            palavrasunicas = self.buscapalavrasunicas(frequencia)
            complete_base = nltk.classify.apply_features(lambda doc: self.extratorpalavras(doc), frasesstemming)
            classificador = nltk.NaiveBayesClassifier.train(complete_base)

        test_stemming = [RSLPStemmer().stem(p) for p in word_tokenize(sentence)]
        new_features = self.extratorpalavras(test_stemming)
        result = classificador.prob_classify(new_features)
        return np.array([result.prob(emotion) for emotion in self.emotions_funcs.keys()])

    def plot_emotion_line_charts(self, scores_list: list, paragraph_end_indices: list, timestamp: str):
        """Plota gráficos de linha para cada emoção ao longo do texto, marcando os finais dos parágrafos.

        Args:
            scores_list (list): Lista de pontuações de emoções.
            paragraph_end_indices (list): Índices que marcam o final de cada parágrafo.
            timestamp (str): Carimbo de data/hora para nomear os arquivos salvos.

        """
        emotions = list(self.emotions_funcs.keys())
        fig, axes = plt.subplots(nrows=len(emotions), ncols=1, figsize=(10, 2 * len(emotions)))
        for i, emotion in enumerate(emotions):
            axes[i].plot([score[i] for score in scores_list], label=f'{emotion} pontuações')
            for end_idx in paragraph_end_indices:
                axes[i].axvline(x=end_idx, color='grey', linestyle='--', label='Fim do Parágrafo' if end_idx == paragraph_end_indices[0] else "")
            axes[i].legend(loc='upper right')
            axes[i].set_title(f'Evolução da Pontuação por Emoção: {emotion}')
            axes[i].set_xlabel('Contagem de frases')
            axes[i].set_ylabel('Pontuações')
        plt.tight_layout()
        plt.savefig(f'generated_html/generated_images/emotion_scores_{timestamp}.png')
        plt.close()

    def plot_pie_chart(self, scores_list: list, timestamp: str):
        """Plota um gráfico de pizza da distribuição de emoções.

        Args:
            scores_list (list): Lista de pontuações de emoções.
            timestamp (str): Carimbo de data/hora para nomear os arquivos salvos.

        """
        labels = list(self.emotions_funcs.keys())
        sizes = [np.mean([score[idx] for score in scores_list]) for idx in range(len(labels))]
        colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen', 'orange']
        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
        plt.title('Proporção de Cada Sentimento')
        plt.savefig(f'generated_html/generated_images/pie_chart_{timestamp}.png')
        plt.close()

    def plot_bar_chart(self, scores_list: list, timestamp: str):
        """Plota um gráfico de barras das frequências de emoções.

        Args:
            scores_list (list): Lista de pontuações de emoções.
            timestamp (str): Carimbo de data/hora para nomear os arquivos salvos.

        """
        labels = list(self.emotions_funcs.keys())
        sizes = [np.sum([score[idx] for score in scores_list]) for idx in range(len(labels))]
        plt.figure(figsize=(10, 6))
        plt.bar(labels, sizes, color='purple', edgecolor='black')
        plt.title('Frequência de Emoções Detectadas')
        plt.xlabel('Emoção')
        plt.ylabel('Frequência')
        plt.savefig(f'generated_html/generated_images/bar_chart_{timestamp}.png')
        plt.close()

    def create_html_visualization(self, timestamp: str, paragraphs: list, sentences: list):
        """Cria uma visualização HTML dos dados analisados, destacando o final de cada frase e incluindo numeração.
    
        Args:
            timestamp (str): Carimbo de data/hora da análise.
            paragraphs (list): Lista de parágrafos do texto analisado.
            sentences (list): Lista de frases do texto analisado.
        """
        base_path = os.path.abspath('generated_html/generated_images')
        pie_chart_path = f'file://{base_path}/pie_chart_{timestamp}.png'
        bar_chart_path = f'file://{base_path}/bar_chart_{timestamp}.png'
        line_chart_path = f'file://{base_path}/emotion_scores_{timestamp}.png'
        
        # Construir o conteúdo HTML do texto analisado com marcações de fim de frase e numeração
        html_paragraphs = []
        sentence_counter = 1
        for paragraph in paragraphs:
            marked_paragraph = paragraph
            for sentence in sentences:
                if sentence in marked_paragraph:
                    marked_paragraph = marked_paragraph.replace(sentence, f'{sentence} <span style="color:red;">[{sentence_counter}]</span>')
                    sentence_counter += 1
            html_paragraphs.append(f'<p>{marked_paragraph}</p>')
    
        html_content = f"""
        <html>
        <head>
            <title>Análise de Sentimentos - Visualização</title>
            <style>
                span {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Texto Analisado</h1>
            <div style='border:1px solid black; padding:10px;'>{''.join(html_paragraphs)}</div>
            <h1>Data e Hora da Análise</h1>
            <div style='border:1px solid black; padding:10px;'>{timestamp}</div>
            <h1>Número de Parágrafos e Frases</h1>
            <div style='border:1px solid black; padding:10px;'>Parágrafos: {len(paragraphs)}, Frases: {len(sentences)}</div>
            <h1>Proporção de Cada Sentimento</h1>
            <img src='{pie_chart_path}' alt='Gráfico de Pizza'>
            <h1>Frequência de Emoções Detectadas</h1>
            <img src='{bar_chart_path}' alt='Gráfico de Barras'>
            <h1>Evolução da Pontuação por Emoção</h1>
            <img src='{line_chart_path}' alt='Gráficos de Linha das Emoções'>
        </body>
        </html>
        """
        with open(f"generated_html/visualization_{timestamp}.html", "w", encoding="utf-8") as file:
            file.write(html_content)

    def open_html_in_browser(self, timestamp: str):
        """Abre o arquivo HTML gerado no navegador padrão.

        Args:
            timestamp (str): Carimbo de data/hora da análise para localizar o arquivo.

        """
        file_path = os.path.abspath(f"generated_html/visualization_{timestamp}.html")
        webbrowser.open(f"file://{file_path}")

    def execute_analysis(self, filepath: str):
        """Executa a análise completa no arquivo especificado.
    
        Args:
            filepath (str): Caminho do arquivo a ser analisado.
    
        """
        paragraphs, sentences = self.process_document(filepath)
        scores_list, paragraph_end_indices = self.analyze_paragraphs(paragraphs)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.plot_emotion_line_charts(scores_list, paragraph_end_indices, timestamp)
        self.plot_pie_chart(scores_list, timestamp)
        self.plot_bar_chart(scores_list, timestamp)
        self.create_html_visualization(timestamp, paragraphs, sentences)
        self.open_html_in_browser(timestamp)

def main():
    os.system('clear')
    print("""
    ====================================
    |      Analisador de Sentimentos   |
    ====================================
    """)
    files = [f for f in os.listdir('test_texts/') if f.endswith('.txt')]
    if not files:
        print("Nenhum arquivo .txt encontrado no diretório atual.")
        return
    print("Selecione o arquivo para análise:")
    for idx, file in enumerate(files):
        print(f"{idx + 1}. {file}")
    choice = int(input("Digite o número do arquivo que deseja analisar: ")) - 1
    if 0 <= choice < len(files):
        filepath = f"test_texts/{files[choice]}"
        if os.path.exists(filepath):
            analyzer = SentimentAnalyzer()
            analyzer.execute_analysis(filepath)
        else:
            print(f"O arquivo selecionado '{filepath}' não foi encontrado.")
    else:
        print("Escolha inválida.")

if __name__ == "__main__":
    main()
