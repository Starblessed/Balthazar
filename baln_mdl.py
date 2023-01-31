import pandas as pd
import statistics as stt
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


# Repara os dados quebrados, separa os dados por espaço e renomeia as ausências de coletas para "Nulo".
def fix_entry(entry):
    fixed_entry0 = entry.replace('P rópria', 'Própria')
    fixed_entry1 = fixed_entry0.replace('/n', ' ')
    fixed_entry = fixed_entry1.replace('Não Realizada', 'Nulo')
    return fixed_entry


# Lista de identificadores dos pontos de coleta.
beach_ids = ['BG000', 'GM000', 'GM001', 'PN000',
             'PS001', 'PS000', 'BD000', 'BD002',
             'BD003', 'BD005', 'BD007', 'BD009',
             'BD010', 'JT000', 'PP010', 'GV001',
             'GV002', 'VD000', 'LB000', 'LB001',
             'LB003', 'IP003', 'IP010', 'IP006',
             'AR000', 'PD000', 'CP100', 'CP004',
             'CP005', 'CP008', 'LM002', 'VR000',
             'UR000', 'BT000', 'BT001', 'FL000', 'FL004']

beach_names = [f'Praia da Barra de Guaratiba {beach_ids[0]}',
               f'Praia do Grumari {beach_ids[1]}',
               f'Praia do Grumari {beach_ids[2]}',
               f'Prainha {beach_ids[3]}',
               f'Praia do Pontal de Sernambetiba {beach_ids[4]}',
               f'Praia do Pontal de Sernambetiba {beach_ids[5]}',
               f'Praia do Recreio {beach_ids[6]}',
               f'Praia do Recreio {beach_ids[7]}',
               f'Praia do Recreio {beach_ids[8]}',
               f'Praia da Barra {beach_ids[9]}',
               f'Praia da Barra {beach_ids[10]}',
               f'Praia da Barra - Pepe {beach_ids[11]}',
               f'Praia da Barra - Quebra-Mar {beach_ids[12]}',
               f'Praia da Joatinga {beach_ids[13]}',
               f'Praia do Pepino {beach_ids[14]}',
               f'Praia de São Conrado {beach_ids[15]}',
               f'Praia de São Conrado {beach_ids[16]}',
               f'Praia do Vidigal {beach_ids[17]}',
               f'Praia do Leblon {beach_ids[18]}',
               f'Praia do Leblon {beach_ids[19]}',
               f'Praia do Leblon {beach_ids[20]}',
               f'Praia de Ipanema {beach_ids[21]}',
               f'Praia de Ipanema {beach_ids[22]}',
               f'Praia de Ipanema {beach_ids[23]}',
               f'Praia do Arpoador {beach_ids[24]}',
               f'Praia do Diabo {beach_ids[25]}',
               f'Praia de Copacabana {beach_ids[26]}',
               f'Praia de Copacabana {beach_ids[27]}',
               f'Praia de Copacabana {beach_ids[28]}',
               f'Praia de Copacabana {beach_ids[29]}',
               f'Praia do Leme {beach_ids[30]}',
               f'Praia Vermelha {beach_ids[31]}',
               f'Praia da Urca {beach_ids[32]}',
               f'Praia de Botafogo {beach_ids[33]}',
               f'Praia de Botafogo {beach_ids[34]}',
               f'Praia do Flamengo {beach_ids[35]}',
               f'Praia do Flamengo {beach_ids[36]}']

# Lista de entrada válidas.
valid_entries = ['Própria', 'Imprópria', 'Nulo']

# Indexador de arquivos.
files = ['d_2017_1', 'd_2017_2', 'd_2018_1', 'd_2018_2', 'd_2019_1', 'd_2019_2']
date_files = ['days_2017', 'days_2018', 'days_2019', 'days_2020']

# Cria um dicionário vazio para armazenar as datas.
dates_0 = {}


# Aplica a função fix_entry(), separa os dados utilizando espaços como separadores, coloca-os em uma lista filtrando
# as entradas válidas.
def clean_data(old_data):
    prep_data = fix_entry(old_data)
    target_list = prep_data.split()
    new_data = []
    for c in target_list:
        if (c in beach_ids) or (c in valid_entries):
            new_data.append(c)
    return new_data


# Cria um dicionário com as praias e atribui cada entrada em ordem cronológica a elas.
def formatter(w_data):
    dict_build = {}
    id_counter = []
    for w_entry in w_data:

        if w_entry in beach_ids:

            if w_entry not in id_counter:
                dict_build.update({w_entry: []})
            id_counter.append(w_entry)
        else:
            dict_build[id_counter[-1]].append(w_entry)

    return dict_build


# Testa os dados formatados pela função formatter() e retorna a quantidade de entradas em cada praia.
# Nota: Para que o programa funcione corretamente, é necessário que todas as praias possuam o mesmo número de entradas.
# Nota 2: Todas as entradas são consideradas. Própria, Imprópria ou Nulo.
def data_validation(data):
    counter_list = []
    indexer = 0
    for checking in beach_ids:
        counter_list.append(len(data[checking]))

    ev = stt.mode(counter_list)
    print(f'-- The expected value is {ev} --')

    for checking in counter_list:
        if checking != ev:
            print(f'{beach_ids[indexer]} - {checking}')
        indexer += 1
    print('---------------------')


# Cria um arquivo .csv (comma separated values) das entradas de cada praia em determinado ano.
def create_csv(dictionary, year):
    df = pd.DataFrame(dictionary)
    df.to_csv(f'raw_data_{year}.csv')


# Recebe um ano, recolhe os dados semestrais e os une, posteriormente aplicando as funções clean_data() e formatter()
# respectivamente. Nota: Esta função retorna os dados finais.
def ftd(year):
    semesters_ref = [f'd_{year}_1', f'd_{year}_2']
    semesters_raw = []
    for reader in semesters_ref:
        temp_file = open(f'{reader}.txt', encoding='utf8')
        semesters_raw.append(temp_file.read())
        temp_file.close()
    cleaned = clean_data(semesters_raw[0]) + clean_data(semesters_raw[1])
    final_data = formatter(cleaned)
    return final_data


# Acessa os arquivos de datas de coleta, retifica as entradas e as armazena em uma lista
def date_formatter(target):
    for reader in date_files:
        temp_file = open(f'{reader}.txt')
        output_text = temp_file.read()
        target.update({f'{reader.replace("days_", "")}': output_text.split()})
        temp_file.close()


# Formata os as datas para o formato de trabalho.
date_formatter(dates_0)


# Função de criação de série temporal. Esta função coloca as datas de coleta mensais em uma única lista indicando
# cada dia de coleta do ano.
def establish_interval(year):
    # Checa se o ano é bissexto e estrutura os meses.
    isleap = False
    if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
        isleap = True

    # Dicionário de meses.
    month_length = {'jan': 31,
                    'feb': 29 if isleap else 28,
                    'mar': 31,
                    'apr': 30,
                    'may': 31,
                    'jun': 30,
                    'jul': 31,
                    'aug': 31,
                    'sep': 30,
                    'oct': 31,
                    'nov': 30,
                    'dec': 31}

    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    formatted = [int(dates_0[f'{year}'][0])]
    actual = 0
    for c in range(1, len(dates_0[f'{year}'])):
        working_month = months[actual]

        if int(dates_0[f'{year}'][c]) < int(dates_0[f'{year}'][c - 1]):
            formatted.append(int(formatted[-1]) + int(dates_0[f'{year}'][c]) + month_length[working_month] - int(
                dates_0[f'{year}'][c - 1]))
            actual += 1

        else:
            formatted.append(int(formatted[-1]) + int(dates_0[f'{year}'][c]) - int(dates_0[f'{year}'][c - 1]))
    return formatted


# Função de estruturação de série temporal binária.
def febin(inputs, dates):
    results = []
    for n in range(len(inputs)):
        mytuple = []
        if inputs[n] == 'Própria':
            mytuple = tuple([dates[n], 1])
        elif inputs[n] == 'Imprópria':
            mytuple = tuple([dates[n], -1])
        elif inputs[n] == 'Nulo':
            mytuple = tuple([dates[n], 0])
        results.append(mytuple)
    return results


# Função de estruturação de série temporal crescente.
def fecres(inputs, dates):
    seq = 0
    results = []
    for n in range(len(inputs)):
        if inputs[n] == 'Própria':
            if seq != 5:
                seq = seq + 1
        elif inputs[n] == 'Imprópria':
            if seq != -5:
                seq = seq - 1
        mytuple = tuple([dates[n], seq])
        results.append(mytuple)
    return results


# Função de estruturação de série temporal crescente adaptada para Mapas de Calor.
def hm_fecres(inputs):
    seq = 0
    results = []
    for n in range(len(inputs)):
        if inputs[n] == 'Própria':
            if seq != 5:
                seq = seq + 1
        if inputs[n] == 'Imprópria':
            if seq != -5:
                seq = seq - 1
        results.append(seq)
    return results


# Função de estruturação de série temporal binária adaptada para Mapas de Calor.
def hm_febin(inputs):
    results = []
    for n in range(len(inputs)):
        if inputs[n] == 'Própria':
            results.append(1)
        elif inputs[n] == 'Imprópria':
            results.append(-1)
        else:
            results.append(0)
    return results


# Funcional, porém incompleta.
def redux(year, dist):
    redux_dict = {}
    date_holder = establish_interval(year)
    if dist == 'bin':
        for beach in beach_ids:
            redux_dict.update({beach: febin(ftd(year)[beach], establish_interval(year))})
    elif dist == 'cres':
        for beach in beach_ids:
            redux_dict.update({beach: fecres(ftd(year)[beach], establish_interval(year))})

    return redux_dict


# Cria um mapa de calor no ano especificado com a função informada.
def baln_heatmap(year, func):
    column_ids = establish_interval(year)
    raw_data = ftd(year)
    my_array = np.zeros(shape=(37, len(ftd(year)['BD010'])))
    df = pd.DataFrame(my_array)
    try:
        if func == 'febin':
            my_data = [hm_febin(raw_data[beach]) for beach in beach_ids]
        elif func == 'fecres':
            my_data = [hm_fecres(raw_data[beach]) for beach in beach_ids]
    except:
        print(f'Invalid function name: {func}')
    contador = 0
    for iterator in my_data:
        df.iloc[contador] = iterator
        contador = contador + 1
    fig = plt.figure()
    fig.set_figwidth(20)
    fig.set_figheight(7)
    plt.title(f'Situação das praias do Rio de Janeiro em {year}')
    sns.heatmap(df, xticklabels=column_ids, yticklabels=beach_ids, cmap='RdYlGn', linewidths=0.05, linecolor='black')
    plt.ylabel('Praia')
    plt.xlabel('Dia do Ano')
    plt.show()


# Cria um gráfico de balneabilidade temporal na praia, no ano e com a função informada.
def plotf(beach: str, year: int, func: str):
    f = plt.figure()
    f.set_figwidth(10)
    if func == 'febin':
        plt.plot(*zip(*febin(ftd(year)[beach], establish_interval(year))))
    elif func == 'fecres':
        plt.plot(*zip(*fecres(ftd(year)[beach], establish_interval(year))))
    else:
        print(f'{func} is not a valid function.')
    plt.xlabel('Dias do ano')
    plt.ylabel('Balneabilidade')
    plt.show()


# Cria um gráfico de balneabilidade histórico da praia e informada com a função especificada.
def hplotf(beach: str, func: str):
    years = [2017, 2018, 2019]
    f = plt.figure()
    f.set_figwidth(10)
    for ct in years:
        if func == 'febin':
            plt.plot(*zip(*febin(ftd(ct)[beach], establish_interval(ct))), label=f'{ct}')
        elif func == 'fecres':
            plt.plot(*zip(*fecres(ftd(ct)[beach], establish_interval(ct))), label=f'{ct}')
        else:
            print(f'{func} is not a valid function.')
            break
    plt.legend(loc='best')
    plt.title(beach_names[beach_ids.index(beach)])
    plt.xlabel('Dias do ano')
    plt.ylabel('Balneabilidade')
    plt.show()
