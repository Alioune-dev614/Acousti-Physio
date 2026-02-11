import os
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import csv
from tkinter.filedialog import askdirectory

from numpy.fft import fft, fftfreq

import wave
import numpy

import unicodedata

import statsmodels.api as sm
from scipy.signal import find_peaks

from scipy.interpolate import InterpolatedUnivariateSpline

from scipy.signal import find_peaks

from scipy import interpolate
import parselmouth

def trouver_regroupement(info_sons, position):
    liste_voyelles = ['a', 'e', 'i', 'o', 'u', 'y', 'A', 'E', 'I', 'O', 'U', 'Y', '@']
    
    nb_voyelles = 0
    nb_consonnes = 0
    v_ou_c = ''
    regroupement = ''
    i = 0
    position_dans_regroupement = 0
    res_position_dans_regroupement = 0
    for son in info_sons:
        
        is_voyelle = False
        
        for voyelle in liste_voyelles:
            
            if voyelle in son[2]:
                if v_ou_c != 'V':
                    nb_voyelles = nb_voyelles +1
                    v_ou_c = 'V'
                    position_dans_regroupement = 1
                else:
                    position_dans_regroupement = position_dans_regroupement +1
                is_voyelle = True
                break
                
        if not is_voyelle:
            if v_ou_c != 'C':
                nb_consonnes = nb_consonnes +1
                v_ou_c = 'C'
                position_dans_regroupement = 1
            else:
                position_dans_regroupement = position_dans_regroupement +1
        
        if i == position:
            if v_ou_c == 'V':
                regroupement = v_ou_c + str(nb_voyelles)
                res_position_dans_regroupement = position_dans_regroupement
            else:
                regroupement = v_ou_c + str(nb_consonnes)
                res_position_dans_regroupement = position_dans_regroupement
        
        i = i+1
        
    if regroupement == 'V1' and nb_voyelles == 1:
        regroupement = 'V'
    if regroupement == 'C1' and nb_consonnes == 1:
        regroupement = 'C'
    
    return regroupement, res_position_dans_regroupement

# export Praat spectrogram (Parselmouth spectrogram object) as greyscale image: enhanced version with preemphasis and bivariate splines smoothing, closer to Praat display
def export_smoothed_grayscale_spectrogram_with_zero_padding(spectrogram, preemphasis_dB_by_octave=6, zero_padding_width_ratio=0, value_for_zero_padding=None):

  X, Y = spectrogram.x_grid(), spectrogram.y_grid()
  Xbins, Ybins = spectrogram.x_bins(), spectrogram.y_bins()
  Xmid = np.column_stack((Xbins[:,0],np.diff(Xbins,axis=1)/2)).sum(axis=1)
  Ymid = np.column_stack((Ybins[:,0],np.diff(Ybins,axis=1)/2)).sum(axis=1)

  # if requested, add pre-emphasis to boost higher frequency display (typical value for pre-emphasis = 6dB/octave)
  if(preemphasis_dB_by_octave>0):
    if(Ymid[0]<=0):
      logYmid = np.append(0,np.log2(Ymid[1::]))
    else:
      logYmid = np.log2(Ymid)
    intensityBoost_dB = (logYmid - min(logYmid))*preemphasis_dB_by_octave
    sg_db = (10 * np.log10(spectrogram.values))+intensityBoost_dB[:,np.newaxis]
  else:
    sg_db = 10 * np.log10(spectrogram.values)

  # interpolate values on both time and frequency dimensions using bivariate splines
  # (adapted from https://stackoverflow.com/a/32567022)
  f = interpolate.RectBivariateSpline(Xmid, Ymid, sg_db.T)
  Xnew = np.linspace(X[0], X[-1], 10*len(X))
  Ynew = np.linspace(Y[0], Y[-1], 10*len(Y))
  sg_bd_interp = f(Xnew, Ynew).T

  # add "zeros" (min value in image unless specified as argument value_for_zero_padding) on the right of the image if zero_padding_width_ratio is over 0 and below 1
  if(zero_padding_width_ratio>0 and zero_padding_width_ratio<1):
    if(value_for_zero_padding==None):
      value_for_zero_padding=np.min(sg_bd_interp)
    n_rows = sg_bd_interp.shape[0]
    n_cols = sg_bd_interp.shape[1]
    n_added_cols = int(np.round(n_cols * zero_padding_width_ratio / (1 - zero_padding_width_ratio)))
    added_zeros = np.ones((n_rows, n_added_cols))*value_for_zero_padding
    zero_padded_sg_bd_interp = np.flipud(np.concatenate((sg_bd_interp,added_zeros), axis = 1))
  else:
    zero_padded_sg_bd_interp = np.flipud(sg_bd_interp)
    
  return zero_padded_sg_bd_interp, sg_db
  
def interpolation_spline(data_num, data):
    # use fitpack2 method
    fnc_interp = InterpolatedUnivariateSpline(data_num, data)
    
    # faire ca pour discretiser l'interpolation:
    interp_num = np.arange(data_num[0], data_num[-1], 1/10)
    interp = fnc_interp(interp_num)
    
    return interp_num, interp
    
def interpolation_spline_create_data_after(data_num, data, nb, freq):
    # use fitpack2 method
    fnc_interp = InterpolatedUnivariateSpline(data_num, data)
    
    # faire ca pour discretiser l'interpolation:
    interp_num = np.arange(data_num[0], data_num[-1] + nb * freq)
    interp = fnc_interp(interp_num)
    
    print(data_num[0])
    print(data_num[-1])
    print(data_num[-1] + nb * freq)
    print(len(data_num))
    print(len(interp_num))
    print(len(interp))
    
    return interp_num, interp

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def decouper_en_mots(i):
    nb_mot = 1
        
    mots = []
        
    # si c'est une phrase
    if "Non_tu_nas_pas_dit_" in i:
            
        nb_mot = 3
        
        ## on coupe les sons 
        partie_droite = i.split("Non_tu_nas_pas_dit_")[1]
            
        premier_mot = partie_droite.split("_")[0]
            
        partie_droite = partie_droite.split("_", 1)[1]
            
        partie_droite = partie_droite.split("quatre_fois_tu_as_dit_")[1]
            
        deuxième_mot = partie_droite.split("_", 1)[0]
            
        partie_droite = partie_droite.split("et_", 1)[1]
            
        troisieme_mot = partie_droite.split("_")[0]
            
        mots.append(premier_mot)
        mots.append(deuxième_mot)
        mots.append(troisieme_mot)
            
    # si c'est un mot
        
    else:
        mots.append(i.split("_")[1])
        
    return nb_mot, mots

def recuperation_dossier_et_infotxt(path):
    # ca on s'en tape
    prime_directory = []
    txt_files = []

    # pour tout sub dossier/fichiers direct
    for fichier in os.listdir(path):
        if '.' not in fichier and 'these' in fichier:
            # c'est le dossier principal
            prime_directory.append(fichier)
        else: 
            if '.txt' in fichier:
                # c'est le fichier principal
                txt_files.append(fichier)
            
    return prime_directory, txt_files

def recuperation_info_sons_depuis_TextGrid(sub_dir_prime, TextGrid, fps):

    nb_son_dans_mot = 0
    info_sons = []

    # on ouvre le TextGrid
    txt = open(os.path.join(sub_dir_prime, TextGrid))
                              
    triplet = []
                                
    # on récupère les infos par sons
    for k, line in enumerate(txt):
        if "intervals: size" in line:
            nb_son_dans_mot = int(line.replace("intervals: size = ", "")) -2
                                        
        if "intervals [" in line:
            info_sons.append(triplet)
            triplet = []
        if "xmin" in line or "xmax" in line or "text" in line:
            line = line.replace("xmin = ", "")
            line = line.replace("xmax = ", "")
            line = line.replace("text = ", "")
            line = line.replace(" ", "")
            line = line.replace("\n", "")
            line = line.replace('"', "")
            triplet.append(line)
                              
    # on retire les indésirables
    info_sons.pop(0)
    info_sons.pop(0)
    txt.close()
                            
    # on récupère la position de ces débuts et fins
    for son in info_sons:
        son[0] = round(float(son[0]) / fps)
        son[1] = round(float(son[1]) / fps)
    
    return nb_son_dans_mot, info_sons

def recuperation_et_coefficientage_donnees(sub_dir_prime, file, coeff, fps):
    try : 
    
        # Read file to get buffer                                                                                               
        ifile = wave.open(os.path.join(sub_dir_prime, file))
        samples = ifile.getnframes()
        audio = ifile.readframes(samples)

        # Convert buffer to float32 using NumPy                                                                                 
        audio_as_np_int16 = numpy.frombuffer(audio, dtype=numpy.int16)
        audio_as_np_float32 = audio_as_np_int16.astype(numpy.float32)

        # Normalise float32 array so that values are between -1.0 and +1.0                                                      
        # max_int16 = 2**15
        max_int16 = float(2 ** (15))
        max_int16 = (max_int16 + 1)
        audio_normalised = audio_as_np_float32 / max_int16
        
        
        data = audio_normalised
        
    except : 
        # recupération des données
        samplerate, data = wavfile.read(os.path.join(sub_dir_prime, file))
        data = data/32786
                        
    # coefficientage des données
    data = [x*coeff for x in data]
                    
    # création de l'absisce 
    data_num = list(range(0, len(data)))
    data_num = [x/fps for x in data_num]
    
    return data_num, data

def fft_data_mot(txt_file, sub_dir_prime, j, phrase, mot, locuteur, graph_path):      
    if "(" in j:
        print("Erreur : \"", j, "\" semble être un doublon")
        print("Poursuite du traitement des autres dossiers...")
        return 
    # recup coeffs et frequence par seconde
    txt = open(txt_file)
    for k, line in enumerate(txt):
        if k == 2:
            fps = 1/float(line)         
        if k == 5:
            Wcoeff = float(line.split("Audio")[1])     
        if k == 6:
            Ocoeff = float(line.split("O-flow")[1])                        
        if k == 7:
            Ncoeff = float(line.split("N-flow")[1])
        if k == 8:
            Acoeff = float(line.split("N-sound")[1])  
            break
    txt.close()
            
    nb_son_dans_mot, info_sons = recuperation_info_sons_depuis_TextGrid(sub_dir_prime, j, fps)

    try:
        # AN
        AN_data_num, AN_data = recuperation_et_coefficientage_donnees(sub_dir_prime, j.replace(".TextGrid", "_AN.wav", 1), Acoeff, fps)
    except:
        pass
    
    try:
        # WAV
        WAV_data_num, WAV_data = recuperation_et_coefficientage_donnees(sub_dir_prime, j.replace(".TextGrid", ".wav", 1), Wcoeff, fps)
    except:
        pass

    try:
        # NAF
        NAF_data_num, NAF_data = recuperation_et_coefficientage_donnees(sub_dir_prime, j.replace(".TextGrid", "_NAF.wav", 1), Ncoeff, fps)
    except:
        pass
        
    try:
        # OAF
        OAF_data_num, OAF_data = recuperation_et_coefficientage_donnees(sub_dir_prime, j.replace(".TextGrid", "_OAF.wav", 1), Ocoeff, fps)
    except:
        pass
    
    try:
        # spectro
        # version Praat (parselmouth)
        max_frequency_spectrogram_Hz = 8000
        duree_max_signal_secondes = 5
        dynamic_range = 70

        sound_path = os.path.join(sub_dir_prime, j.replace(".TextGrid", ".wav", 1))
        snd=parselmouth.Sound(sound_path)

        spectro=snd.to_spectrogram(maximum_frequency=max_frequency_spectrogram_Hz)
        
        #zeroPaddingWidthRatio = (duree_max_signal_secondes-snd.get_total_duration())/duree_max_signal_secondes
        zeroPaddingWidthRatio = 0
        
        zero_padded_sg_bd_interp, sg_db = export_smoothed_grayscale_spectrogram_with_zero_padding(spectro, preemphasis_dB_by_octave=6, zero_padding_width_ratio=zeroPaddingWidthRatio)
    except:
        pass
        
    # plot
    
    try:
        fig0, axs0 = plt.subplots(figsize=[13,4.30])
        axs0.imshow(zero_padded_sg_bd_interp, aspect="auto", cmap="gray_r", vmin=sg_db.max() - dynamic_range, vmax=None)
        fig0.savefig(os.path.join(graph_path, mot + "_spectro.png"), dpi=200)
    except:
        pass
    
    try:
        fig1, axs1 = plt.subplots(figsize=[13,4.30])
        axs1.set_title("WAV")
        axs1.plot(WAV_data_num, WAV_data, lw='1')
        axs1.set_xlabel(r"Temps (s)")
        axs1.set_ylabel(r"Débit (L/s)")
        axs1.set_xlim(0,WAV_data_num[-1])
        for son in info_sons[:-1]:
            axs1.axvline(x=WAV_data_num[son[1]], color = 'black', linestyle = '-', lw='1')
        fig1.savefig(os.path.join(graph_path, mot + "_SA.png"), dpi=200)
    except:
        pass
    
    try:
        fig2, axs2 = plt.subplots(figsize=[13,4.30])
        axs2.set_title("NAF et OAF")
        axs2.axhline(y = 0, color = 'black', linestyle = '--', lw='1')
        axs2.plot(NAF_data_num, NAF_data, label='NAF', lw='1')
        axs2.plot(OAF_data_num, OAF_data, label='OAF', lw='1')
        axs2.set_xlabel(r"Temps (s)")
        axs2.set_ylabel(r"Débit (L/s)")
        axs2.legend()
        axs2.set_xlim(0,NAF_data_num[-1])
        for son in info_sons[:-1]:
            axs2.axvline(x=NAF_data_num[son[1]], color = 'black', linestyle = '-', lw='1')
        fig2.savefig(os.path.join(graph_path, mot + "_flow.png"), dpi=200)
    except:
        pass
    
    try:
        fig3, axs3 = plt.subplots(figsize=[13,4.30])
        axs3.set_title("AN")
        axs3.plot(AN_data_num, AN_data, lw='1')
        axs3.set_xlabel(r"Temps (s)")
        axs3.set_ylabel(r"Débit (L/s)")
        axs3.set_xlim(0,AN_data_num[-1])
        for son in info_sons[:-1]:
            axs3.axvline(x=AN_data_num[son[1]], color = 'black', linestyle = '-', lw='1')
        fig3.savefig(os.path.join(graph_path, mot + "_SAN.png"), dpi=200)
    except:
        pass
        
    fig, axs = plt.subplots(4, 2, width_ratios=[4, 1], figsize=[12.20,10.80])
    
    try:
        axs[0,0].imshow(zero_padded_sg_bd_interp, aspect="auto", cmap="gray_r", vmin=sg_db.max() - dynamic_range, vmax=None)
    except:
        pass
    
    try:
        axs[1,0].set_title("WAV")
        axs[1,0].plot(WAV_data_num, WAV_data, lw='1')
        axs[1,0].set_xlabel(r"Temps (s)")
        axs[1,0].set_ylabel(r"Débit (L/s)")
        axs[1,0].set_xlim(0,WAV_data_num[-1])
        for son in info_sons[:-1]:
            axs[1,0].axvline(x=WAV_data_num[son[1]], color = 'black', linestyle = '-', lw='1')
    except:
        pass
    
    try:
        axs[2,0].set_title("NAF et OAF")
        axs[2,0].axhline(y = 0, color = 'black', linestyle = '--', lw='1')
        axs[2,0].plot(NAF_data_num, NAF_data, label='NAF', lw='1')
        axs[2,0].plot(OAF_data_num, OAF_data, label='OAF', lw='1')
        axs[2,0].set_xlabel(r"Temps (s)")
        axs[2,0].set_ylabel(r"Débit (L/s)")
        axs[2,0].legend()
        axs[2,0].set_xlim(0,NAF_data_num[-1])
        for son in info_sons[:-1]:
            axs[2,0].axvline(x=NAF_data_num[son[1]], color = 'black', linestyle = '-', lw='1')
    except:
        pass
    
    try:
        axs[3,0].set_title("AN")
        axs[3,0].plot(AN_data_num, AN_data, lw='1')
        axs[3,0].set_xlabel(r"Temps (s)")
        axs[3,0].set_ylabel(r"Débit (L/s)")
        axs[3,0].set_xlim(0,AN_data_num[-1])
        for son in info_sons[:-1]:
            axs[3,0].axvline(x=AN_data_num[son[1]], color = 'black', linestyle = '-', lw='1')
    except:
        pass
    
    ind_son = 0
    # on exploite
    for son in info_sons:
    
        if son[1] - son[0] < 1024:
            if ind_son == 0:
                son[1] = son[0] + 1024
            elif ind_son == len(info_sons) -1:
                son[0] = son[1] - 1024
            else:
                son[0] = int(son[0] - (1024 - son[1] + son[0])/2)
                if son[0] < 0:
                    son[0] = 0
                son[1] = son[0] + 1024
        
        try:
            # (2)
            AN_auto = sm.tsa.acf(AN_data[son[0]:son[1]], nlags=2000)
            AN_peaks = find_peaks(AN_auto)[0] # Find peaks of the autocorrelation
            AN_lag = AN_peaks[0] # Choose the first peak as our pitch component lag
            AN_pitch = 10000 / AN_lag # Transform lag into frequency
            AN_pitch = 1/AN_pitch
            
            AN_x = np.array(AN_data[son[0]:son[1]])
            
            AN_N = AN_x.size

            # (3)
            # AN_milieu = int(AN_N/2)
            AN_debut_milieu = int(AN_N/2 - AN_pitch*1.5)
            AN_fin_milieu = AN_debut_milieu + int(AN_pitch * 3)
            
            ## (0) => on agrandit pour faire au moins 1024 de taille (qui est le minimum)
            if AN_fin_milieu - AN_debut_milieu < 1024:
                i = 0
                nbr = 1024 
                while nbr > 0:
                    i = i+1
                    nbr = nbr -AN_pitch
                AN_debut_milieu = int(AN_N/2 - AN_pitch*(i/2))
                AN_fin_milieu = AN_debut_milieu + int(AN_pitch * i)
            
            AN_fenetre = AN_x[AN_debut_milieu:AN_fin_milieu]
            AN_N_fenetre = len(AN_fenetre)

            # (4)
            AN_fd=np.multiply(np.hamming(AN_N_fenetre),AN_fenetre)
            AN_fd2= np.abs(fft(AN_fd))
            AN_fftMag = AN_fd2[0:512]
            AN_dBMag = 20*numpy.log10(AN_fftMag/2e-2)
            
            # (5)
            # AN_fftFreqs = fftfreq(AN_N, d=fps)
            AN_fftFreq = 1/AN_N_fenetre*20000
            
            AN_fftFreqs = list(range(512))
            AN_fftFreqs = [x * AN_fftFreq for x in AN_fftFreqs]
            
            ## lissage FFT
            AN_liss_x, AN_liss_y = interpolation_spline(AN_fftFreqs, AN_dBMag.tolist())
        except:
            pass
        
        try:
        # (2)
            WAV_auto = sm.tsa.acf(WAV_data[son[0]:son[1]], nlags=2000)
            WAV_peaks = find_peaks(WAV_auto)[0] # Find peaks of the autocorrelation
            WAV_lag = WAV_peaks[0] # Choose the first peak as our pitch component lag
            WAV_pitch = 10000 / WAV_lag # Transform lag into frequency
            WAV_pitch = 1/WAV_pitch
            
            WAV_x = np.array(WAV_data[son[0]:son[1]])
            
            WAV_N = WAV_x.size

            # (3) => ramener en seconde le f0
            # WAV_milieu = int(WAV_N/2)
            WAV_debut_milieu = int(WAV_N/2 - WAV_pitch*1.5)
            WAV_fin_milieu = WAV_debut_milieu + int(WAV_pitch * 3)
            
            ## (0) => on agrandit pour faire au moins 1024 de taille (qui est le minimum)
            if WAV_fin_milieu - WAV_debut_milieu < 1024:
                i = 0
                nbr = 1024 
                while nbr > 0:
                    i = i+1
                    nbr = nbr -WAV_pitch
                WAV_debut_milieu = int(WAV_N/2 - WAV_pitch*(i/2))
                WAV_fin_milieu = WAV_debut_milieu + int(WAV_pitch * i)
            
            WAV_fenetre = WAV_x[WAV_debut_milieu:WAV_fin_milieu]
            WAV_N_fenetre = len(WAV_fenetre)

            # (4)
            WAV_fd=np.multiply(np.hamming(WAV_N_fenetre),WAV_fenetre)
            
            WAV_fd2= np.abs(fft(WAV_fd))
            WAV_fftMag = WAV_fd2[0:512]
            WAV_dBMag = 20*numpy.log10(WAV_fftMag/2e-2)
            
            # (5)
            # WAV_fftFreqs = fftfreq(WAV_N, d=fps)
            WAV_fftFreq = 1/WAV_N_fenetre*20000
            
            WAV_fftFreqs = list(range(512))
            WAV_fftFreqs = [x * WAV_fftFreq for x in WAV_fftFreqs]
            
            ## lissage FFT
            WAV_liss_x, WAV_liss_y = interpolation_spline(WAV_fftFreqs, WAV_dBMag.tolist())
        except:
            pass
        
        
        
        
        
        if ind_son >= 4:
            print("Erreur : ", mot, " dépasse 4 sons, au moins 1 fft de perdu dans le fichier global")
        
        else:
            axs[ind_son,1].set_title("FFT")
            try:
                axs[ind_son,1].plot(AN_liss_x, AN_liss_y, label='Nasal', lw='1')
            except:
                pass
            try:
                axs[ind_son,1].plot(WAV_liss_x, WAV_liss_y, label='Oral', lw='1')
            except:
                pass
            try:
                peaks = find_peaks(AN_liss_y[1000:], prominence=5, distance=1250)
                ind_marker = 1
                for peak_ind in peaks[0][:4]:
                    px = AN_liss_x[peak_ind +1000]
                    py = AN_liss_y[peak_ind +1000]
                    axs[ind_son,1].plot(px, py+5, marker="$"+str(ind_marker)+"$", markersize=5, markeredgecolor="#1f77b4", markerfacecolor="none")
                    ind_marker = ind_marker +1
            except:
                pass
            try:
                peaks = find_peaks(WAV_liss_y[1000:], prominence=5, distance=1250)
                ind_marker = 1
                for peak_ind in peaks[0][:4]:
                    px = WAV_liss_x[peak_ind +1000]
                    py = WAV_liss_y[peak_ind +1000]
                    axs[ind_son,1].plot(px, py+5, marker="$"+str(ind_marker)+"$", markersize=5, markeredgecolor="#ff7f0e", markerfacecolor="none")
                    ind_marker = ind_marker +1
            except:
                pass
            axs[ind_son,1].set_xlabel(r"Fréquence (Hz)")
            axs[ind_son,1].set_ylabel(r"Amplitude $X(f)$")
            axs[ind_son,1].legend()
            axs[ind_son,1].set_xlim(0,2000)
            axs[ind_son,1].set_ylim(-20,95)
        
        figfft, axfft = plt.subplots(figsize=[13,13])
        
        axfft.set_title("FFT")
        try:
            axfft.plot(AN_liss_x, AN_liss_y, label='Nasal', lw='1')
        except:
            pass
        try:
            axfft.plot(WAV_liss_x, WAV_liss_y, label='Oral', lw='1')
        except:
            pass
        try:
            peaks = find_peaks(AN_liss_y[1000:], prominence=5, distance=1250)
            ind_marker = 1
            for peak_ind in peaks[0][:4]:
                px = AN_liss_x[peak_ind +1000]
                py = AN_liss_y[peak_ind +1000]
                axfft.plot(px, py+5, marker="$"+str(ind_marker)+"$", markersize=5, markeredgecolor="#1f77b4", markerfacecolor="none")
                ind_marker = ind_marker +1
        except:
            pass
        try:
            peaks = find_peaks(WAV_liss_y[1000:], prominence=5, distance=1250)
            ind_marker = 1
            for peak_ind in peaks[0][:4]:
                px = WAV_liss_x[peak_ind +1000]
                py = WAV_liss_y[peak_ind +1000]
                axfft.plot(px, py+5, marker="$"+str(ind_marker)+"$", markersize=5, markeredgecolor="#ff7f0e", markerfacecolor="none")
                ind_marker = ind_marker +1
        except:
            pass
        axfft.set_xlabel(r"Fréquence (Hz)")
        axfft.set_ylabel(r"Amplitude $X(f)$")
        axfft.legend()
        axfft.set_xlim(0,2000)
        axfft.set_ylim(-20,95)
        
        regroupement, position_dans_regroupement = trouver_regroupement(info_sons, info_sons.index(son))
        
        if position_dans_regroupement > 1:
            figfft.savefig(os.path.join(graph_path, mot + "_" + regroupement + "(" + str(position_dans_regroupement) + ")" + ".png"), dpi=200)
        else:
            figfft.savefig(os.path.join(graph_path, mot + "_" + regroupement + ".png"), dpi=200)

        ind_son = ind_son+1
        
    fig.savefig(os.path.join(graph_path, mot + ".png"), dpi=200)
    plt.close('all')
    
    # si il manquait le text grid remplir de trous     
    if len(info_sons) == 0:
        print('- ', mot, ' pas de TextGrid')

def fft_data_locuteur(path, locuteur):

    for dir_patient_temoins in os.listdir(os.path.join(path, locuteur)):
    
        if "FINAL" in dir_patient_temoins or "postop" in dir_patient_temoins or "preop" in dir_patient_temoins:
        
            if "FINAL" in dir_patient_temoins:
                path_final = os.path.join(path, locuteur, dir_patient_temoins)
            if "postop" in dir_patient_temoins or "preop" in dir_patient_temoins:
                path_final = os.path.join(path, locuteur, dir_patient_temoins, "FINAL")
            
            prime_directorys, txt_files = recuperation_dossier_et_infotxt(path_final)
            
            for prime_directory in prime_directorys:
                
                for file in txt_files:
                    if prime_directory in file:
                        txt_file = file
         
                # on construit le chemin du info txt 
                path_txt = os.path.join(path_final, txt_file)   
                
                # on construit le chemin du dossier principal
                path_prime = os.path.join(path_final, prime_directory)

                # on prend toutes les phrases
                for phrase in os.listdir(path_prime):

                    if '.' not in phrase and "RESULTAT" not in phrase:
                    
                        phrase_corrigee = phrase.split("_", 1)[1].replace("_", " ")
                        
                        nb_mots, mots = decouper_en_mots(phrase)
                        
                        if len(mots) != 3 or (mots[0] != '' and mots[1] != '' and mots[2] != ''):
                        
                            sub_dir_prime = os.path.join(path_prime, phrase)
                            
                            graph_directory = os.path.join(sub_dir_prime, "graphes")
                            
                            try:
                                os.mkdir(graph_directory)
                            except:
                                pass
                            
                            # si c'est une voyelle 
                            
                            for mot in mots:
                                
                                presence_TextGrid = False
                                
                                # on cherche nos fichiers dans le lot 
                                for fichier in os.listdir(sub_dir_prime):
                                    
                                    fichier2 = strip_accents(fichier)
                                    mot2 = strip_accents(mot)
                                    
                                    # extrait infos pour les mot ayant un TextGrid
                                    if ".TextGrid" in fichier and ("_" + mot2) in fichier2:
                                        textgrid = fichier
                                        fft_data_mot(path_txt, sub_dir_prime, textgrid, phrase_corrigee, mot, locuteur, graph_directory)
                                        presence_TextGrid = True
                                    else:
                                        
                                        try:
                                            spt = fichier.split('m.', -1)
                                            fichier3 = spt[0] + 'n.' + spt[1]
                                            fichier4 = strip_accents(fichier3)
                                            
                                            if ".TextGrid" in fichier3 and ("_" + mot2) in fichier4:
                                                textgrid = fichier
                                                fft_data_mot(path_txt, sub_dir_prime, textgrid, phrase_corrigee, mot, locuteur, graph_directory)
                                                presence_TextGrid = True
                                        except:
                                            pass
                                            
                                            
                                        try:
                                            spt = fichier.split('in.', -1)
                                            fichier3 = spt[0] + 'ain.' + spt[1]
                                            fichier4 = strip_accents(fichier3)
                                            
                                            if ".TextGrid" in fichier3 and ("_" + mot2) in fichier4:
                                                textgrid = fichier
                                                fft_data_mot(path_txt, sub_dir_prime, textgrid, phrase_corrigee, mot, locuteur, graph_directory)
                                                presence_TextGrid = True
                                        except:
                                            pass
                                            
                                        try:
                                            spt = fichier.split('m.', -1)
                                            fichier3 = spt[0] + 'me.' + spt[1]
                                            fichier4 = strip_accents(fichier3)
                                            
                                            if ".TextGrid" in fichier3 and ("_" + mot2) in fichier4:
                                                textgrid = fichier
                                                fft_data_mot(path_txt, sub_dir_prime, textgrid, phrase_corrigee, mot, locuteur, graph_directory)
                                                presence_TextGrid = True
                                        except:
                                            pass
                                            
                                        try:
                                            spt = fichier.split('im.', -1)
                                            fichier3 = spt[0] + 'ain.' + spt[1]
                                            fichier4 = strip_accents(fichier3)
                                            
                                            if ".TextGrid" in fichier3 and ("_" + mot2) in fichier4:
                                                textgrid = fichier
                                                fft_data_mot(path_txt, sub_dir_prime, textgrid, phrase_corrigee, mot, locuteur, graph_directory)
                                                presence_TextGrid = True
                                        except:
                                            pass
                                
                                if not presence_TextGrid:
                                    print('- ', mot, ' pas de TextGrid')
                                    
                        else:
                            print("Erreur : \"", phrase_corrigee, "\" est une phrase non supportée.")
                            print("Poursuite du traitement des autres dossiers...")

def fft_data_all_locuteur():

    # notre chemin
    path = askdirectory()

    # pour chaque locuteur 
    for locuteur in os.listdir(path):
      
        if '.' not in locuteur:
            
            # affichage de la progression
            print(locuteur)        
            fft_data_locuteur(path, locuteur)
 
# execution de tout à l'appel du fichier
fft_data_all_locuteur()