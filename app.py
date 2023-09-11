
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def medidor_de_similaridade(text1, text2):
    # Instancia do contado de n-grams(combinações de palavras que ocorrem juntas, normalmente com uma certa frequência)
    count = CountVectorizer(analyzer = 'word', ngram_range = (1, 1))
    # Cria um dicionario com um codigo para cada palavra
    vocab2int = count.fit([text1, text2]).vocabulary_
    # Cria uma matriz de n_grams
    n_grams = count.fit_transform([text1, text2])
    # Cria um array com a contagem de quantas vezes aquele codigo aparece no texto 1 e quantas vezes no texto 2
    n_grams_array = n_grams.toarray()
    # Pega o menor numero do indice dos dois arrays
    intercection = np.amin(n_grams.toarray(), axis=0)
    # Soma as intersecções
    sum = np.sum(intercection)
    # Soma os nums do array do texto 1
    Index_A = 0
    count = np.sum(n_grams.toarray()[Index_A])
    # Porcentagem igual
    porcentEqual = (sum/count) * 100

    return(porcentEqual)


texto1 = "Condicionador Elseve 500 ml"
texto2 = "Shampoo 400 ml Elseve"

porcent = medidor_de_similaridade(texto1, texto2)

print(porcent)