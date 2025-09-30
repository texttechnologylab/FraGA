# Helper.py

Helper Functions to conduct the `hand_movement.py` and `Pink_Noise_hand_movement.py` analysis.

## Functions

- **Loading LoopVarIds** from TXT-Files  
- **Loading Tracking Data** from JSON-Files  
- **Generating a Dominant Hand Dictionary** (stores dominant hand for each `playerid`)  
- **Merging Dictionaries**  
- **Extracting Timestamps** for all `playerid`s:  
  - **Dialogue** `extract_dialogue_timestamps`: Timestamps for all sentences from Dialogue File  
  - **Hesitations** `extract_bracketed_timestamps`: Timestamps for all hesitations from word_timestamps_json  
  - **Speechacts** `extract_speechact_timestamps`: Timestamps for all sentences ending with a dot (.) or question mark (?)  

---

## To test the different Timestamp Extracting Methods

**Requirements:**  
- Dialogue Folder – Transcription Chat (Player 1 + Player 2)  
- Crisper_Whisper_Json – Word_timestamp  

**SET PATHS at line:**  
- 312: Dialogue Folder: Transcriptions Chat  
- 313: Crisper_Whisper_Json: Word-Timestamps  

---

# Pink_Noise_hand_movement.py

**Requirements:**  
- LoopVarIDs  
- Handleft.json  
- Handright.json  
- Dialogue Folder – Transcription Chat (Player 1 + Player 2)  
- Crisper_Whisper_Json – Word_timestamps  

**To run the Analysis: SET PATHS at line:**  
- 144: `tracking_json_zip` – Tracking Data JSON-Files  
- 147: Dialogue-Folder – Transcriptions Chat  
- 148: Crisper_Whisper_Json – Word-Timestamps  

**Output:**  
- 167: Set Output Path for generated Exceltable  
- 208: Set Output Path for generated plot [Left Hand]  
- 242: Set Output Path for generated plot [Right Hand]  

**Used PlayerIds for Original Analysis (59):**  
See: `FraGa/scripts/results/Pinknoise Dialog.xlsx`

---

# hand_movement.py

**Requirements:**  
- Excel-table – `FraGa_allPlayer_playerID.xlsx`  
- LoopVarIDs  
- Handleft.json  
- Handright.json  
- Dialogue Folder – Transcription Chat (Player 1 + Player 2)  
- Crisper_Whisper_Json – Word_timestamps  

**To run the Analysis: SET PATHS at line:**  
- 298: Set Output Path for generated plot  
- 382: Set Path to Excel-table `FraGa_allPlayer_playerID.xlsx`  
- 385: Dialogue-Folder – Transcriptions Chat  
- 386: Crisper_Whisper_Json – Word-Timestamps  
- 395: LoopVarId  
- 396: `tracking_json_zip` – Tracking Data JSON-Files

---

# settings.py

Sets the variables for different data input and output paths for the eye movement analysis.
Set variables before using the eye analysis functions.

---

# utils.py

Helper Functions to conduct the `eye_visual.py` analysis.

**Requirements:**  
- Excel-table – `FraGa_allPlayer_playerID.xlsx`

## Functions

- **Loading LoopVarIds** from TXT-Files  
- **Loading Tracking Data** from JSON-Files  
- **Importing data from the Excel sheet into a dictionary, with the option to create a json file**

---

# eye_visual.py

Main module for eye analysis functions

**Requirements:**  
- Excel-table – `FraGa_allPlayer_playerID.xlsx`  
- LoopVarIDs  
- body.json  
- eye.json
- head.json
- Dialogue Folder – Transcription Chat (Player 1 + Player 2)  
- Crisper_Whisper_Json – Word_timestamps 

## Functions

- **data_process()**: Analyses the view direction of all players with existing data and returns plots for the results.
- **load_id_qick_3d_plot()**: Creates 3d plots to visualize view directions for all players with existing data. View calculations differ slightly from the data_process function, since a dynamic calculation is needed for the 3d plot.
