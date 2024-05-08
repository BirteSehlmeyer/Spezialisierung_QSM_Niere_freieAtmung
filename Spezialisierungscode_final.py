import numpy as np
import os
import pydicom
import shutil
import nibabel as nib
import subprocess
from matplotlib import pyplot as plt
import matlab.engine #importieren der Libary
import tkinter as tk
from tkinter import filedialog

########################################################################################################################################
################################## Funktionen  #########################################################################################
########################################################################################################################################

def erstelle_ordner(dicom_pfad:str, ordner_name:str)->str:
    # erstellt einen neuen Ordner, wenn er noch nicht existiert
    # Parameter: - dicom_pfad: String, Pfad zum DICOM-Ordner
               # - ordner_name: String, Name des neuen Ordners
    # Rückgabe: - ordner_pfad: String, Pfad zum erstellten oder vorhandenen Ordner
    ordner_pfad = os.path.join(dicom_pfad, ordner_name)
    if not os.path.exists(ordner_pfad):
        os.makedirs(ordner_pfad)
        print(f"Der Ordner '{ordner_pfad}' wurde erfolgreich erstellt.")
    else:
        print(f"Der Ordner '{ordner_pfad}' existiert bereits.")
    return ordner_pfad


def sortiere_dicom_dateien(dicom_pfad:str, magnitude_pfad:str, phase_pfad:str):
     # sortiert DICOM-Dateien nach Magnituden- und Phasenbildern
     # Parameter:
            # - dicom_pfad: String, Pfad zum DICOM-Ordner
            # - magnitude_pfad: String, Pfad zum Magnituden-Ordner
            # - phase_pfad: String, Pfad zum Phasen-Ordner
    for filename in os.listdir(dicom_pfad):
        file_pfad = os.path.join(dicom_pfad, filename)
        try:
            ds = pydicom.dcmread(file_pfad)
            if hasattr(ds, 'ImageType') and len(ds.ImageType) > 2:
                tag_value = ds.ImageType[2]
                if tag_value == 'P':
                    P_folder_path = erstelle_ordner(dicom_pfad, 'Phase')
                    if len(os.listdir(P_folder_path)) != 104:
                        shutil.copy(file_pfad, phase_pfad)
                        print(f"Die Datei '{filename}' wurde nach '{phase_pfad}' kopiert.")
                    else:
                        print(f"Der Ordner '{phase_pfad}' enthält bereits 104 Elemente. Datei '{filename}' nicht kopiert.")
                else:
                    M_folder_path = erstelle_ordner(dicom_pfad, 'Magnitude')
                    if len(os.listdir(M_folder_path)) != 104:
                        shutil.copy(file_pfad, magnitude_pfad)
                        print(f"Die Datei '{filename}' wurde nach '{magnitude_pfad}' kopiert.")
                    else:
                        print(f"Der Ordner '{magnitude_pfad}' enthält bereits 104 Elemente. Datei '{filename}' nicht kopiert.")
            else:
                print(f"ImageType-Tag fehlt oder ungültig in: {file_pfad}")
        except pydicom.errors.InvalidDicomError:
            print(f"Ungültige Dicom-Datei: {file_pfad}")
        except Exception as e:
            print(f"Fehler beim Verarbeiten von {file_pfad}: {e}")


def lese_erste_dicom_datei(ordner_pfad:str):
    # liest die erste DICOM-Datei aus einem Ordner und speichert die Daten in einem pydicom.Dataset
    # Parameter:- ordner_pfad: String, Pfad zum Ordner, der die DICOM-Datei enthält
    # Rückgabe: - ds: pydicom.Dataset, das die gelesenen DICOM-Daten enthält
    # überprüft, ob der Ordner existiert
    if not os.path.exists(ordner_pfad):
        print(f"Der Ordner '{ordner_pfad}' existiert nicht.")
        return None
    # Liste der Dateien im Ordner
    dateien_im_ordner = os.listdir(ordner_pfad)
    # überprüft, ob Dateien im Ordner vorhanden sind
    if not dateien_im_ordner:
        print(f"Keine Dateien im Ordner '{ordner_pfad}' gefunden.")
        return None
    # wählt die erste Datei aus der Liste aus
    erste_datei = dateien_im_ordner[0]
    datei_pfad = os.path.join(ordner_pfad, erste_datei)

    try:
        ds = pydicom.dcmread(datei_pfad)
        return ds
    except pydicom.errors.InvalidDicomError:
        print(f"Ungültige Dicom-Datei: {datei_pfad}")
        return None
    except Exception as e:
        print(f"Fehler beim Lesen von {datei_pfad}: {e}")
        return None


def lese_TE_dicom_dateien(verzeichnis_pfad:str, TE:list)->list:
    # liest DICOM-Dateien im angegebenen Verzeichnis, extrahiert EchoTime und gibt die verschiedenen Echozeiten jeweils einmal zurück
    # Parameter: - verzeichnis_pfad: String, Pfad zum Verzeichnis mit DICOM-Dateien
    #            - TE: Liste zur Speicherung von Echozeiten
    # Rückgabe: - unterschiedliche_TE: Liste der verschiedenen Echozeiten

    # Schleife durch die DICOM-Dateien im Verzeichnis
    for index, element in enumerate(os.listdir(verzeichnis_pfad)):
        # V_dat: Daten des jeweiligen Verzeichnisses, entweder der Magnitude oder der Phase 
        V_dat = pydicom.dcmread(os.path.join(verzeichnis_pfad, element)) # DICOM-Datei einlesen + Pfade zusammenführen
        TE.append(V_dat.EchoTime)

    #set(), um Echzeiten nur einmal in liste zu erhalten
    unterschiedliche_TE_string_list = list(set(TE))
    unterschiedliche_TE_string_list.sort()
    unterschiedliche_TE = [float(element) for element in unterschiedliche_TE_string_list] # unterschiedliche_TE in ms
    return unterschiedliche_TE


def verarbeite_dicom_daten(data_array:np.array, verzeichnis_pfad:str, echo_time:float, unterschiedliche_TE:list)->np.array:
    # verarbeitet DICOM-Daten für eine bestimmte Echozeit und aktualisiert das Datenarray.
    # Parameter:
    # - data_array: NumPy-Array, das zu aktualisierende Datenarray
    # - verzeichnis_pfad: String, Pfad zum Verzeichnis mit DICOM-Dateien
    # - echo_time: Float, aktuelle Echozeit
    # - unterschiedliche_TE: Liste der verschiedenen Echozeiten
    # Rückgabe:
    # - data_array: aktualisiertes Datenarray.
    dicom_dateien_fuer_te = []
    # Schleife durch die DICOM-Dateien im jeweiligen verzeichnis_pfad
    for element in os.listdir(verzeichnis_pfad):
        V_dat = pydicom.dcmread(os.path.join(verzeichnis_pfad, element)) # DICOM-Datei einlesen + Pfade zusammenführen
        # überprüft, ob die Echozeit der aktuellen Datei mit der aktuellen Echozeit übereinstimmt
        if V_dat.EchoTime == echo_time:
            dicom_dateien_fuer_te.append(V_dat)  # fügt die DICOM-Datei zur Liste hinzu

    # verarbeitet die Liste der DICOM-Dateien mit der aktuellen Echozeit
    for index, V_dat in enumerate(dicom_dateien_fuer_te):
        V_arr = V_dat.pixel_array # Umwandlung in ein zweidimensionales Array
        data_array[:, :, index, unterschiedliche_TE.index(echo_time)] = V_arr # schreibt Grauwerte in das vierdimensionale data_arrray

    return data_array


def MRTSignal(N:int, T:int)->np.array:
    # erstellt vierdimensionales Array
    # Parameter:
    # - N: Anzahl der Schichten
    # - T: Anzahl der verschiedenen Echozeiten
    # Rückgabe:
    # - komplexMRT_array
    komplexMRT_array = np.zeros(data_shape, dtype = complex)
    # mithilfe einer geschachtelten Schleife durch die Anzahl der aufgenommenen Schichten N und die T verschiedenen Echozeiten wird das Array 
    # mit den komplexen Werten der MRT-Signale gefüllt
    for n in range(N):
        for t in range(T):
            komplexMRT_array[:, :, n , t] = data_M[:, :, n, t] * (np.cos(scaled_data_P[:, :, n, t]) + 1j * np.sin(scaled_data_P[:, :, n, t]))
            # für jedes Indexpaar wird aus den Magnitudendaten data_M und den Phasendaten scaled_data das komplexe MRT-Signal berechnet
    return komplexMRT_array


def save_nifti(data:np.array, folder_path:str, file_name:str):
    # Funktion zum Abspeichern von NIfTI-Dateien
    # Parameter:
    # - data: Datenarray
    # - folder_path: String, Ordnerpfad
    # - file_name: String, Dateiname
    nifti_data = data[:, :, :]  # extrahiert Daten für die erste Echozeit [:, :, :, 0]
    nifti = nib.Nifti1Image(nifti_data, affine=np.eye(4))
    # speichert das NIfTI-Objekt als Datei
    file_path = os.path.join(folder_path, f"{file_name}.nii")
    nib.save(nifti, file_path)


########################################################################################################################################
################################## ORDNER ERSTELLEN UND DICOM-DATEIEN SORTIEREN ########################################################
########################################################################################################################################

root = tk.Tk()
root.withdraw()  # versteckt das Hauptfenster

custom_title = "Wähle Ordner der DICOM-Dateien aus"
root.wm_title(custom_title)  
dicom_pfad = filedialog.askdirectory(title=custom_title)  # öffnet den Dialog für die Auswahl eines Ordners

if dicom_pfad:
    print(f"Gewählter Ordner: {dicom_pfad}")
else:
    print("Kein Ordner ausgewählt.")

# Der Name der neuen Unterordner im DICOM-Ordner
Ordnername_Phase = 'Phase'
Ordnername_Magnitude = 'Magnitude'


P_folder_path = os.path.join(dicom_pfad, Ordnername_Phase)
M_folder_path = os.path.join(dicom_pfad, Ordnername_Magnitude)

sortiere_dicom_dateien(dicom_pfad, M_folder_path, P_folder_path)     

# element for element in os.listdir(M_folder_path) iteriert durch die Liste von M_folder_path 
# if os.path.isfile(os.path.join(M_folder_path, element) behandelt nur die Elemente, für die es wahr ist, also keine Verzeichnisse oder andere Elemente
# os.path.join(M_folder_path, element) erstellt den vollständigen Pfad zum aktuellen Element (Datei) in der Schleife
# wenn os.path.isfile() wahr ist, wird die Datei zur Liste Magnitudenordner hinzugefügt
Magnitudenordner = [element for element in os.listdir(M_folder_path) if os.path.isfile(os.path.join(M_folder_path, element))]



########################################################################################################################################
################################## BILDEIGENSCHAFTEN DER ERSTEN DATEI UND DIE ECHOZEITEN AUSLESEN ######################################
########################################################################################################################################
    
ds = lese_erste_dicom_datei(M_folder_path) # oder P_folder_path

TE = []
unterschiedliche_TE = lese_TE_dicom_dateien(M_folder_path, TE) # enthält nur unterschiedliche Echozeiten
print(unterschiedliche_TE)
T = len(unterschiedliche_TE) # T = 4, da 4 verschiedenene Echozeiten
N = int(len(Magnitudenordner)/T) #Anzahl der aufgenommenen Schichten = 26


Z = ds.AcquisitionMatrix[0] #256
S = ds.AcquisitionMatrix[3] #208
# 26 Schichten für 4 Echozeiten = 104

# Größe der vier dimensionalen Arrays
data_shape = (S, Z, N, T)


# Erstellen der leeren vierdimensionalen Arrays data_M und data_P
data_M = np.zeros(data_shape)
data_P = np.zeros(data_shape)

voxel_size_string_list = list(ds.PixelSpacing)
voxel_size_string_list.append(ds.SliceThickness)
resonance_frequency_string = ds.ImagingFrequency
resonance_frequency = float(resonance_frequency_string)*1e6 # 1e6, da MHz

voxel_size =  [float(element) for element in voxel_size_string_list]
print(type(voxel_size))


########################################################################################################################################
################################## ALLE DICOM-DATEIEN AUSLESEN UND VERARBEITEN #########################################################
########################################################################################################################################


for echo_time in unterschiedliche_TE:
    data_M = verarbeite_dicom_daten(data_M, M_folder_path, echo_time, unterschiedliche_TE)
    data_P = verarbeite_dicom_daten(data_P, P_folder_path, echo_time, unterschiedliche_TE)


new_min = -np.pi
new_max = np.pi

K = 2**12 # 12-Bit-Wertebereich, d.h. 2^12 - 1 = 4095

scaled_data_P= (np.interp(data_P, (0,K-1), (new_min, new_max))) *(-1) # invertiert Skala, da negative Phasen in Brain Suite hell dargestellt werden

komplexMRTSignal = MRTSignal(N, T) # befüllt Array mit den komplexen MRT-Signalen, die sich aus den Phasen- und Magnituden-Infomationen berechnen laassen


eng = matlab.engine.start_matlab() # startet die Matlab Engine

#################################################################################################################################################################
############################################# ENTFALTUNG DER PHASE (PHASE UNWRAPPING) ###########################################################################
#################################################################################################################################################################

eng.cd(r'C:\Users\sehbir01\Documents\Matlab-Skripte\MEDI_toolbox_eb\functions\_spurs_gc', nargout=0)

unterschiedliche_TE = [x * 0.001 for x in unterschiedliche_TE] # Umrechnung von ms in s
unterschiedliche_TE = [round(x, 4) for x in unterschiedliche_TE] # Runden auf die 4. Nachkommastelle für äquidistante Echozeitabstände (Delta TE)
print(unterschiedliche_TE)

# Umrechnung in Datentyp, der in MATLAB-Skripte eingelesen werden kann
komplexMRTSignal_mlb = matlab.double(komplexMRTSignal,is_complex = True)
unterschiedliche_TE_mlb = matlab.double(unterschiedliche_TE)
resonance_frequency_mlb = matlab.double(resonance_frequency)
voxel_size_mlb = matlab.double(voxel_size)

wwater, wfat, wfreq, wunwph_uf, unwphw, N_std = eng.spurs_gc(komplexMRTSignal_mlb, unterschiedliche_TE_mlb, resonance_frequency_mlb, voxel_size_mlb, nargout=6)
#nargout= Anzahl der Outputvariablen
#output:
#	- wwater: Wasserkarte
#	- wfat:   Fettkarte
#	- wfreq:  Frequenzkarte in rad nach T2*-IDEAL als fine tunning --> Input für weitere QSM-Skripte
#	- wunwph_uf:  Feldkarte nach Entfaltung und Entfernung der chemischen Verschiebung, Initialwert für T2*-IDEAL
#	- unwphw: Ergebnis der Phasenentfaltung
	
#input:
#	- komplexMRTSignal: komplexe MRT-Daten
#	- unterschiedliche_TE: Liste der verschiedenen Echozeiten
#	- resonance frequency: Resonanzfrequenz 
#	- voxel_size: Voxelgröße

wfreq_phaseunwrapping_nparray = np.array(wfreq._data).reshape(wfreq.size, order ='F')
# wandelt Daten in wfreq in np.array um und formatiert es spaltenweie in neue Form (order = 'F' Fortran-Style)  

root.withdraw()  # versteckt das Hauptfenster

custom_title = "Wähle Ordner, in dem Plots-Ordner erstellt werden sollen"
root.wm_title(custom_title) 
plots_folder_path = filedialog.askdirectory(title=custom_title) # öffnet den Dialog für die Auswahl des Ordners, in dem Plots-Ordner gespeichert werden soll


if plots_folder_path:
    print(f"Gewählter Ordner: {plots_folder_path}")
else:
    print("Kein Ordner ausgewählt.")

Ordnername_Plots = "Plots_png"
P_folder_path = erstelle_ordner(plots_folder_path, Ordnername_Plots)


# Der vollständige Pfad des neuen Ordners
new_plots_folder_path = os.path.join(plots_folder_path, Ordnername_Plots)

# überprüft, ob der Ordner bereits existiert, falls nicht wird er erstellt
if not os.path.exists(new_plots_folder_path):
    os.makedirs(new_plots_folder_path)
    print(f"Ordner '{Ordnername_Plots}' wurde im Pfad '{plots_folder_path}' erstellt.")
else:
    print(f"Der Ordner '{Ordnername_Plots}' existiert bereits im Pfad '{plots_folder_path}'.")



plt.figure()
img=plt.imshow(wfreq_phaseunwrapping_nparray[:, :, 4]/np.pi, cmap='gray')
#plt.colorbar(label='Phase in rad')
cbar = plt.colorbar(img, label='Entfaltete Phase [rad]')
cbar.set_ticks([0.5, 1, 1.5, 2, 2.5, 3])
cbar.set_ticklabels(['π/2', 'π', '3/2 π', '2 π', '5/2 π', '3 π'])  # Tickbeschriftungen
#plt.colorbar.set_label('Phase (Rad)')
#plt.xlabel('256')
#plt.ylabel('208')
plt.title('Resultat nach Phase-Unwrapping')
entfaltete_phase_name = '\\entfaltetePhase.png'
plt.savefig(new_plots_folder_path + entfaltete_phase_name) 
#plt.legend()
plt.show()
plt.close()


###################################################################################################################################################################
########################################################## VORBEREITUNG DER HINTERGRUNDFELD-ENTFERNUNG ############################################################
###################################################################################################################################################################

################################# Vorberitung des Inputs für die Maskenerstellung mit der Software Brain Suite ####################################################

diffusion_value = 26 # Wert des Anisotropischen Diffusionsfilters, Gebiete geringen Kontrastes werden geglättet und welche mit hohem Kontrast bleiben unverändert
edge_values = 0.55 # Kanten-Konstante legt Dicke fest, ab der Kante als solche gilt

root.withdraw()  # Versteckt das Hauptfenster
custom_title = "Wähle Ordner, in dem Nifti-Ordner erstellt werden soll"
root.wm_title(custom_title) 
nifti_folder_path = filedialog.askdirectory(title=custom_title) # öffnet den Dialog für die Auswahl des Ordners, in dem Nifti-Ordner erstellt werden soll
# erstellt einen Ordner, um die NIfTI-Dateien zu speichern


if nifti_folder_path:
    print(f"Gewählter Ordner: {nifti_folder_path}")
else:
    print("Kein Ordner ausgewählt.")

Ordnername_NIFTIs = "nifti-Dateien"
new_nifti_folder_path = erstelle_ordner(nifti_folder_path, Ordnername_NIFTIs) # vollständiger Pfad des neuen Ordners

# überprüft, ob der Ordner bereits existiert, falls nicht wird er erstellt
if not os.path.exists(new_nifti_folder_path):
    os.makedirs(new_nifti_folder_path)
    print(f"Ordner '{Ordnername_NIFTIs}' wurde im Pfad '{nifti_folder_path}' erstellt.")
else:
    print(f"Der Ordner '{Ordnername_NIFTIs}' existiert bereits im Pfad '{nifti_folder_path}'.")


save_nifti(data_P, new_nifti_folder_path, "phase")
save_nifti(wfreq_phaseunwrapping_nparray, new_nifti_folder_path, "unwrapping_phase")


# speichert Magnituden-Nifti zur Maskenerstellung ab
magnitude_data_TE1 = data_M[:,:,:, 0] # 3D Array der ersten Echozeit, da am signalreichsten
magnitude_nifti = nib.Nifti1Image (magnitude_data_TE1, affine=np.eye(4))  
file_magnitude_nii = f"magnitude.nii"  # Name der NIfTI-Datei
file_path_magnitude_nii = os.path.join(new_nifti_folder_path, file_magnitude_nii)
nib.save(magnitude_nifti, file_path_magnitude_nii)


# Dateiauswahl für die Eingabedatei magnitude.nii
custom_title = "Wähle Datei magnitude.nii aus"
root.wm_title(custom_title)  # Set custom title
input_magnitude = filedialog.askopenfilename(title=custom_title) 


if input_magnitude:
    print(f"Ausgewählte Eingabedatei: {input_magnitude}")
else:
    print("Keine Eingabedatei ausgewählt.")


neuer_datei_name = "mask.nii" # neuer Name für das Duplikat
file_path_mask_nii = os.path.join(new_nifti_folder_path, neuer_datei_name) # Ziel-Pfad für das Duplikat


if not os.path.exists(file_path_magnitude_nii):
    print(f"Die Datei '{file_path_magnitude_nii}' existiert nicht.")
else:
    try:
        # dupliziert die Datei
        shutil.copy2(file_path_magnitude_nii, file_path_mask_nii)
        print(f"Die Datei wurde erfolgreich dupliziert und umbenannt: {file_path_mask_nii}")
    except Exception as e:
        print(f"Fehler beim Duplizieren und Umbenennen der Datei: {e}")


# Dateiauswahl für die Ausgabedatei mask.nii
custom_title = "Wähle Datei mask.nii aus"
root.wm_title(custom_title) 
output_mask = filedialog.asksaveasfilename(title=custom_title, defaultextension=".nii", filetypes=[("NIfTI files", "*.nii")])  

if output_mask:
    print(f"Ausgewählte Ausgabedatei: {output_mask}")
else:
    print("Keine Ausgabedatei ausgewählt.")

################## final Maskenerstellung mit dem Tool Brain Surface Extractor (BSE) der Software Brain Suite ####################################################

command = '"C:\\Program Files\\BrainSuite19a\\bin\\bse.exe" -i ' + input_magnitude + ' -o ' + output_mask + ' -r 1 -d ' +  str(diffusion_value) + ' -s ' + str(edge_values) + ' -p'

try:
# Aufruf des externen Programms
    result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, text=True)
    # Zeige die Ausgabe an
    print("Programmausgabe:", result.stdout)

except subprocess.CalledProcessError as e:
    print(f"Fehler beim Ausführen des Programms: {e}")

# überprüft, ob die Ausgabedatei existiert
if not os.path.exists(output_mask):
    print("Die Maske wurde nicht erfolgreich erstellt.")
    exit()

# lädt die NIfTI-Datei
file_mask_nii = f"mask.nii"  # Name der NIfTI-Datei
file_path_mask_nii = os.path.join(new_nifti_folder_path, file_mask_nii)
nifti_mask = nib.load(file_path_mask_nii)
nifti_mask_data = nifti_mask.get_fdata()

# fragt Schwellenwert ab (Ab welchem Wert sollen Maskenwerte auf 1 gesetzt werden?)
threshold_value = float(input("Bitte gib den Schwellenwert für die Maske ein, (Standardwert 0): "))

# setzt alle Maskenwerte größer als Schwellenwert auf 1
modified_mask_data = np.where(nifti_mask_data > threshold_value, 1, 0)

# erstellt ein neues NIfTI-Objekt mit den geänderten Daten
modified_mask_nifti = nib.Nifti1Image(modified_mask_data, nifti_mask.affine, nifti_mask.header)

# speichert die modifizierte NIfTI-Datei
modified_mask_name = f"modifizierte_maske.nii"  # Name der modifizierten NIfTI-Datei
modified_mask_path = os.path.join(new_nifti_folder_path, modified_mask_name)
nib.save(modified_mask_nifti, modified_mask_path)

rotated_modified_mask_data = np.rot90(modified_mask_data, k=-1)  # k=-1 für gegen den Uhrzeigersinn 
rotated_modified_mask_nifti = nib.Nifti1Image(rotated_modified_mask_data, modified_mask_nifti.affine, modified_mask_nifti.header)

# speichert die rotierte modifizierte NIfTI-Datei
rotated_modified_mask_name = f"rotierte_modifizierte_maske.nii"  # Name der modifizierten rotierten NIfTI-Datei
rotated_modified_mask_path = os.path.join(new_nifti_folder_path, rotated_modified_mask_name)
nib.save(rotated_modified_mask_nifti, rotated_modified_mask_path)

# extrahiert das Datenarray aus rotated_modified_mask_nifti
rotated_modified_mask_data = rotated_modified_mask_nifti.get_fdata()
# Schicht des Datenarrays, um die gewünschte Ebene zu erhalten (z. B. Schicht 14)
slice_rotated_modified_mask = rotated_modified_mask_data[:, :, 13]
slice_rotated_modified_mask = np.rot90(slice_rotated_modified_mask, k=1) # Schicht gegen den Uhrzeigersinn drehen 

plt.imshow(slice_rotated_modified_mask, cmap='gray')
plt.title('Maske')
plt.colorbar(label='Intensität [a.u.]')
plt.axis() 
maske_name = '\\Maske.png'
plt.savefig(new_plots_folder_path + maske_name) 
plt.show()
plt.close()


# überprüft, ob die Formen der Daten übereinstimmen
if modified_mask_data.shape != wfreq_phaseunwrapping_nparray.shape:
    raise ValueError("Die Formen der Maske und des Bildes stimmen nicht überein.")

# verrechnt die Pixelwerte elementweise
multiplication_data = np.multiply(modified_mask_data, wfreq_phaseunwrapping_nparray) 
# erstellt ein neues NIfTI-Objekt mit den verrechneten Daten
multiplication_nifti = nib.Nifti1Image(multiplication_data, modified_mask_nifti.affine, modified_mask_nifti.header)

# speichert das Ergebnis in einer neuen NIfTI-Datei
multiplication_name = f"Multiplikation.nii" 
multiplication_path = os.path.join(new_nifti_folder_path, multiplication_name)
nib.save(multiplication_nifti, multiplication_path)


#################################################################################################################################################################
################################################################ HINTERGRUNDFELD-ENTFERNUNG #######################################################################
#################################################################################################################################################################

# Maske in Float
modified_mask_data = modified_mask_data.astype(np.float64)

total_field = matlab.double(wfreq_phaseunwrapping_nparray)
Mask = matlab.double(modified_mask_data)

matrix_size = matlab.double((S, Z, N))


# Benutzer wählt zwischen Verarbeitungsschritten aus
auswahl = input('Wähle Background fiel removal (LBV, V_SHARP_2d, V_SHARP_3d, V_SHARP_beide): ')

if auswahl == 'LBV':
    print('LBV wird durchgeführt.')
    eng.cd(r'C:\Users\sehbir01\Documents\Matlab-Skripte\MEDI_toolbox_eb\functions\_LBV', nargout=0)
    local_field = eng.LBV(total_field, Mask, matrix_size, voxel_size_mlb, nargout = 1)
    #fL = eng.LBV(iFreq, Mask, matrix_size, voxel_size_mlb, tol, depth, peel, N1, N2, N3, nargout = 1)
    background_field_removal = np.array(local_field._data).reshape(local_field.size, order ='F')
    img=plt.imshow(background_field_removal[:,:,13]/np.pi, cmap='gray')
    cbar = plt.colorbar(img, label='Phase des lokalen Felds [rad]')
    cbar.set_ticks([-2, -1 , 0, 1, 2])
    cbar.set_ticklabels(['-2π', '-π', '0', 'π', '2π'])  # Tickbeschriftungen
    plt.title('Resultat nach Background Field Removal mit LBV')
    plt.axis()
    Phaselokal_name = '\\Phaselokal_LBV.png'
    plt.savefig(new_plots_folder_path + Phaselokal_name) 
    plt.show()
    plt.close()

# Output:
#	local_field - lokales Feld 

# Input:
#	total_field - Gesamtes Feld
#	Mask - ROI
#	matrix_size - Matrixgröße 
#	voxel_size - Voxelgröße 
#	tol - Abbruchkriterium der Iterationen --> niedrige Toleranz: höhere Genauigkeit, mehr Rechenleistung notwendig
	
#	Nachfolgende Inputs sind nicht zwingend erforderlich und haben Default-Werte im Matlab-Skript vorgegeben
#	depth - Anzahl der Längenskalen, für die größte Längenskala gilt 2^depth * voxel size.
#	peel - Anzahl der Grenzschichten 
#	N1 - Iterationen für jede Tiefe vor dem rekursiven Aufruf
#	N2 - Iterationen für jede Tiefe nach dem rekursiven Aufruf
#	N3 - Iterationen auf der feinsten Skala, nachdem FMG (full multigrid algorithm) beendet ist


elif auswahl == 'V_SHARP_2d':
    print('V_SHARP_2d wird durchgeführt.')
    smvsize = 12 #default
    padsize = matlab.double((12, 12, 12)) #default

    eng.cd(r'C:\Users\sehbir01\Documents\Matlab-Skripte\STISuite_V3.0\Core_Functions_P', nargout=0)
    local_field = eng.V_SHARP_2d(total_field,Mask,'voxelsize',voxel_size_mlb,'padsize',padsize,'smvsize',smvsize)
    background_field_removal = np.array(local_field._data).reshape(local_field.size, order ='F')

    #plt.imshow(background_field_removal[:,:,13], cmap='gray')
    img=plt.imshow(background_field_removal[:,:,13]/np.pi, cmap='gray')
    #plt.colorbar(label='Phase in rad')
    cbar = plt.colorbar(img, label='Phase des lokalen Felds [rad]')
    cbar.set_ticks([-2, -1 , 0, 1, 2])
    cbar.set_ticklabels(['-2π', '-π', '0', 'π', '2π'])  # Tickbeschriftungen
    plt.title('Resultat nach Background Field Removal mit V_SHARP_2d')
    plt.axis()
    Phaselokal_name = '\\Phaselokal_V_SHARP_2d.png'
    plt.savefig(new_plots_folder_path + Phaselokal_name) 
    plt.show()
    plt.close()


elif auswahl == 'V_SHARP_3d':
    print('V_SHARP_3d wird durchgeführt.')
    smvsize = 12 #default
    padsize = matlab.double((12, 12, 12)) #default

    eng.cd(r'C:\Users\sehbir01\Documents\Matlab-Skripte\STISuite_V3.0\Core_Functions_P', nargout=0)
    local_field = eng.V_SHARP(total_field,Mask,'voxelsize',voxel_size_mlb,'smvsize',smvsize)
    background_field_removal = np.array(local_field._data).reshape(local_field.size, order ='F')

    #plt.imshow(background_field_removal[:,:,13], cmap='gray')
    img=plt.imshow(background_field_removal[:,:,13]/np.pi, cmap='gray')
    #plt.colorbar(label='Phase in rad')
    cbar = plt.colorbar(img, label='Phase des lokalen Felds [rad]')
    cbar.set_ticks([-2, -1 , 0, 1, 2])
    cbar.set_ticklabels(['-2π', '-π', '0', 'π', '2π'])  # Tickbeschriftungen
    plt.title('Resultat nach Background Field Removal mit V_SHARP_3d')
    plt.axis()
    Phaselokal_name = '\\Phaselokal_V_SHARP_3d.png'
    plt.savefig(new_plots_folder_path + Phaselokal_name) 
    plt.show()
    plt.close()


elif auswahl == 'V_SHARP_beide':
    print('V_SHARP_beide wird durchgeführt.')
    smvsize = 12 #default
    padsize = matlab.double((12, 12, 12)) #default

    eng.cd(r'C:\Users\sehbir01\Documents\Matlab-Skripte\STISuite_V3.0\Core_Functions_P', nargout=0)
    local_field_2d = eng.V_SHARP_2d(total_field,Mask,'voxelsize',voxel_size_mlb,'padsize',padsize,'smvsize',smvsize)

    eng.cd(r'C:\Users\sehbir01\Documents\Matlab-Skripte\STISuite_V3.0\Core_Functions_P', nargout=0)
    local_field = eng.V_SHARP(local_field_2d,Mask,'voxelsize',voxel_size_mlb,'smvsize',smvsize)

    background_field_removal = np.array(local_field._data).reshape(local_field.size, order ='F')
   
    img=plt.imshow(background_field_removal[:,:,13]/np.pi, cmap='gray')
    cbar = plt.colorbar(img, label='Phase des lokalen Felds [rad]')
    cbar.set_ticks([-2, -1 , 0, 1, 2])
    cbar.set_ticklabels(['-2π', '-π', '0', 'π', '2π'])  # Tickbeschriftungen
    plt.title('Resultat nach Background Field Removal mit V_SHARP_beide')
    plt.axis() 
    Phaselokal_name = '\\Phaselokal_V_SHARP_beide.png'
    plt.savefig(new_plots_folder_path + Phaselokal_name) 
    plt.show()
    plt.close()


else:
    # Fehlermeldung für ungültige Auswahl
    raise ValueError('Ungültige Auswahl. Bitte LBV, V_SHARP_2d oder V_SHARP_beide eingeben.')


save_nifti(background_field_removal, new_nifti_folder_path, "Magnetfeldverzerrung_lokal")

print('Läuft')

#########################################################################################################################################################
################################################################ INVERSES PROBLEM #######################################################################
#########################################################################################################################################################


B_direction = matlab.double([0, 0, 1]) # Feldrichtung # stimmt so nur, wenn wir streng transversal messen, sonst Drehung berechnen
B0 = 3.0 # Feldstärke in Tesla 


Delta_TE = round(unterschiedliche_TE_mlb._data[1]-unterschiedliche_TE_mlb._data[0], 4) * 1000 # Umrechnung in ms

NewMask = Mask # ggf. mit V-SHARP anpassen, um Randpixel auch nicht zur Maske zu zählen
padsize = matlab.double((12, 12, 12)) # Default

Susceptibility_mlb = eng.QSM_star(local_field,NewMask,'TE', Delta_TE,'B0',B0,'H',B_direction,'padsize',padsize,'voxelsize',voxel_size_mlb, nargout = 1)

# Output:
#	Susceptibility: finale QSM-Daten
	
# Inputs:
#	local_field: lokales Feld nach der Hintergrundfeld-Korrektur
#	NewMask: Neue Maske, kann ggf. angepasst werden
#	Delta_TE: Echozeitenabstand
#	B0: B0 Feldstärke in Tesla
#	B_direction: Feldrichtung
#	padsize: Größe für Padarray zur Erhöhung der numerischen Genauigkeit
#	voxelsize: Voxelgröße

Susceptibility = np.array(Susceptibility_mlb._data).reshape(Susceptibility_mlb.size, order ='F')

plt.imshow(Susceptibility[:,:,13], cmap='gray')
plt.title('Suszeptibilitätskarte')
plt.colorbar(label='Suszeptibilität [ppm]')
plt.axis() 
Suszeptibilitätskarte_name = '\\Suszeptibilitätskarte.png'
plt.savefig(new_plots_folder_path + Suszeptibilitätskarte_name) 
#plt.savefig(r'C:\Users\sehbir01\Desktop\Plots für Institutmeetings\Suszeptibilitätskarte.png') 
plt.show()
plt.close()


# Der vollständige Pfad des neuen Ordners
#new_nifti_sus_folder_path = os.path.join(nifti_sus_folder_path, nifti_sus_folder)
nifti_suszeptibility = np.fliplr(np.rot90(Susceptibility.data,k=1))


# erstellt ein neues NIfTI-Objekt mit den geänderten Daten
suszeptibilität_nifti = nib.Nifti1Image(nifti_suszeptibility, affine =np.eye(4))

suszeptibilität_name = f"Suszeptibilität.nii" 
suszeptibilität_path = os.path.join(new_nifti_folder_path, suszeptibilität_name)
nib.save(suszeptibilität_nifti, suszeptibilität_path)



A=1
