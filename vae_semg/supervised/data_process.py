import os
import time
import numpy as np
import pandas as pd
from scipy import signal



#--------------------- Procesamiento de datos de archivos txt a matrices numpy ---------------------#
# leer txt
def txt2array(txt_path):
    """
    :param txt_path:    ruta específica de un solo archivo txt
    :return:            matriz preprocesada de dimensión 2 <class 'np.ndarray'> del archivo txt de entrada
    """
    table_file = pd.read_table(txt_path, sep=' ', header=None)
    txt_file = table_file.iloc[:, :]
    txt_array = txt_file.values

    return txt_array

    return txt_array

# leer txt 
def txt2array0(txt_path):
    """
    :param txt_path:    ruta específica de un solo archivo txt
    :return:            matriz preprocesada de dimensión 2 <class 'np.ndarray'> del archivo txt de entrada
    """
    table_file = pd.read_table(txt_path, header=None)
    txt_file = table_file.iloc[:, :]
    txt_array = txt_file.values

    return txt_array

# leer txt 
def txt2array_revise(txt_path):
    """
    :param txt_path:    ruta específica de un solo archivo txt
    :return:            matriz preprocesada de dimensión 2 <class 'np.ndarray'> del archivo txt de entrada
    """
    table_file = pd.read_table(txt_path, header=None)
    # txt_file = table_file.iloc[:, :] #
    txt_file = table_file.iloc[:, :-1]
    txt_array = txt_file.values

    return txt_array


# --------------------- Preprocesamiento de datos EMG --------------------#
def preprocessing(data):
    """
    :param data:    8*400 emg data <class 'np.ndarray'>    400*8
    :return:        data instance after rectifying and filter  8*400
    """
    # scalar
    data = 2 * (data + 128) / 256 - 1
    # rectify
    data_processed = np.abs(data)
    # transpose (400, 8) -> (8, 400)
    data_processed = np.transpose(data_processed)
    # filter
    wn = 0.05
    order = 4
    b, a = signal.butter(order, wn, btype='low')
    data_processed = signal.filtfilt(b, a, data_processed)      # data_processed <class 'np.ndarray': 8*400>
    # return (4, 800)
    return data_processed       # <class 'np.ndarray'> 4*800


# --------------------- Detección de la región de actividad muscular --------------------#
def detect_muscle_activity(emg_data):
    """
    :param      emg_date: 8 channels of emg data -> 8*400
    :return:
                index_start: indice de inicio de la región de activación muscular
                index_end:   índice final de la región de activación muscular
    """
    # plot emg_data
    # plt.plot(emg_data.transpose())
    # plt.show()
    fs = 200        # frecuencia de muestreo
    min_activation_length = 50  # longitud mínima de la región de activación muscular (en puntos de muestra)
    num_frequency_of_spec = 50  # número de puntos de frecuencia en el espectrograma
    hamming_window_length = 25  # longitud de la ventana de hamming
    overlap_samples = 10    # número de puntos de superposición entre ventanas adyacentes
    threshold_along_frequency = 18  # umbral a lo largo de la dimensión de frecuencia para detectar la región de activación muscular
    sumEMG = emg_data.sum(axis=0)   # sum 8 channel data into one vector
    # plt.plot(sumEMG)
    # plt.show()
    f, time, Sxx = signal.spectrogram(sumEMG, fs=fs, window='hamming', nperseg=hamming_window_length, noverlap=overlap_samples,
                                   nfft=num_frequency_of_spec, detrend=False, mode='complex')
    # 43.6893
    # test plot
    Sxx = Sxx * 43.6893
    # spec_values = abs(Sxx)
    # plt.pcolormesh(time, f, spec_values, shading='gouraud')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()
    # plt.plot(sumEMG)
    # plt.show()
    spec_values = abs(Sxx)  # obtener el valor absoluto del espectrograma
    spec_vector = spec_values.sum(axis=0)   # obtener el vector espectral sumando a lo largo de la dimensión de frecuencia
    # plt.plot(spec_vector)
    # plt.show()
    indicated_vector = np.zeros(shape=(spec_vector.shape[0] + 2),) # crear un vector indicador con dos elementos adicionales
    # establecer los elementos del vector indicador según el umbral
    for index, element in enumerate(spec_vector):
        if element > threshold_along_frequency:
            indicated_vector[index+1] = 1
    # print('indicated_vector: %s' % str(indicated_vector))
    # print('indicated_vector.shape: %s' % str(indicated_vector.shape))
    index_greater_than_threshold = np.abs(np.diff(indicated_vector)) # calcular la diferencia absoluta entre elementos adyacentes
    if index_greater_than_threshold[-1] == 1:
        index_greater_than_threshold[-2] = 1
    # Eliminar el último elemento
    index_greater_than_threshold = index_greater_than_threshold[:- 1]
    # obtener los índices donde la diferencia es 1
    index_non_zero = np.where(index_greater_than_threshold == 1)[0]
    # calcular los índices de las muestras correspondientes en la señal EMG original
    index_of_samples = np.floor(fs * time - 1)
    num_of_index_non_zero = index_non_zero.shape[0]
    # calcular la longitud de la señal EMG
    length_of_emg = sumEMG.shape[0]
    # print('length of emg : %f points' % length_of_emg)
    # determinar los índices de inicio y fin de la región de activación muscular
    if num_of_index_non_zero == 0:
        index_start = 1
        index_end = length_of_emg
    elif num_of_index_non_zero == 1:
        index_start = index_of_samples[index_non_zero]
        index_end = length_of_emg
    else:
        index_start = index_of_samples[index_non_zero[0]]
        index_end = index_of_samples[index_non_zero[-1] - 1]
    # ampliar un poco la región de activación muscular
    num_extra_samples = 25
    index_start = max(1, index_start - num_extra_samples)
    index_end = min(length_of_emg, index_end + num_extra_samples)
    # si la longitud de la región de activación muscular es demasiado corta, considerar toda la señal como región de activación muscular
    if (index_end - index_start) < min_activation_length:
        index_start = 0
        index_end = length_of_emg - 1
    # print(index_start)
    # print(index_end)
    # return spec_vector, time, spec_values
    return index_start, index_end


# --------------------- asignar etiquetas a los datos EMG --------------------#
def label_indicator(path):
    label = None
    if 'relax' in path: # mano relajada
        label = 0
    elif 'fist' in path: # puño cerrado
        label = 1
    elif 'fingersSpread' in path: # dedos extendidos
        label = 2
    elif 'doubleTap' in path: # doble toque
        label = 3
    elif 'waveIn' in path: # ola hacia adentro
        label = 4
    elif 'waveOut' in path: # ola hacia afuera
        label = 5
    return label


# --------------------- extracción de características del espectrograma --------------------#
# calcular el espectrograma de un vector EMG
def cal_spectrogram_vector(vector, fs=200, npserseg=57, noverlap=0):
    frequencies_samples, time_segment_sample, spectrogram_of_vector = signal.spectrogram(x=vector, fs=fs,
                                                                                         nperseg=npserseg,
                                                                                         noverlap=noverlap,
                                                                                         window='hann',
                                                                                         scaling='spectrum')
    return spectrogram_of_vector, time_segment_sample, frequencies_samples

# calcular el espectrograma de un conjunto de datos EMG
def calculate_spectrogram(dataset):
    """
    :param dataset: (189, (8, 52))
    :return: dataset_spectrogram: (189, (4, 8, 14))
    """
    dataset_spectrogram = [] # inicializar la lista para almacenar los espectrogramas de todo el conjunto de datos
    # calcular el espectrograma para cada ejemplo en el conjunto de datos
    for examples in dataset:    # examples -> (8, 52)
        canals = [] # inicializar la lista para almacenar los espectrogramas de cada canal
        # calcular el espectrograma para cada canal
        for electrode_vector in examples:   # electrode -> (52, )
            spectrogram_of_vector, time_segment_sample, frequencies_samples = \
                cal_spectrogram_vector(electrode_vector, npserseg=28, noverlap=20)
            # spectrogram_of_vector <ndarray: (15, 4)>
            # remover frecuencias bajas inutiles del sEMG (0-5Hz)
            spectrogram_of_vector = spectrogram_of_vector[1:]   # spectrogram_of_vector <ndarray: (14, 4)>
            canals.append(np.swapaxes(spectrogram_of_vector, 0, 1))     # canals (8, (4, 14))
        # intercambiar las dimensiones de los canales y el tiempo
        example_to_classify = np.swapaxes(canals, 0, 1)     # example_to_classify <tuple: (4, 8, 14)>
        dataset_spectrogram.append(example_to_classify)
    # retornar el conjunto de datos de espectrogramas
    return dataset_spectrogram


# --------------------- extracción de características MAV --------------------#
# MAV: Músculo de Activación Media
def mav(emg_data):
    """
    :param emg_data:    <class 'np.ndarray'>    (8, 40)
    :return:            mav feature vector of the input emg matrix     (8, )
    """
    mav_result = np.mean(abs(emg_data), axis=1)
    return mav_result