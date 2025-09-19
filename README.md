[![Paper](http://img.shields.io/badge/paper-SemDial-B31B1B.svg)](https://www.semdial.org/anthology/Z25-Luecking_semdial_3316.pdf)
[![Conference](http://img.shields.io/badge/conference-SemDial--2025-4b44ce.svg)](https://semdial2025.github.io/)
[![version](https://img.shields.io/github/license/texttechnologylab/FraGA)]()

# FraGA
Head and hand movements during turn transitions: data-based multimodal analysis using the Frankfurt VR Gesture–Speech Alignment Corpus


# Abstract
We introduce FRAGA, a VR-based corpus of direction giving dialogues following the model of the SaGA corpus. 
The tracking data of FRAGA are used to carry out multimodal computing: 
we look at turn transitions and re-evaluate findings from the literature on realworld dialogues and compare them with avatar-mediated VR dialogues. 
Interestingly, the established temporal overlap patterns of turns could not be replicated, 
and no significant amount of partner-directed head orientation (approximating gaze) was observed around turn transition points. 
However, the special status of hesitations is evidenced by the co-occurring head movements, but not by hand movements. 
In addition, we apply pink noise distribution fitting to the dialogue data, in particular to the hand movements. 
Here we find that hand movements indeed follow 1⁄f fluctuations, a property of “meta-stable” dynamic systems.

# Data

## FraGa_allPlayer_playerID.xlsx

* Date: Date of the experiment.
* Time: Schedule time of the experiment.
* Person: Role of the participant (Person 1: Router, Person 2: Follower).
* Gender: Gender of the participant.
* Age: Age of the participant.
* DominantHand: Dominant hand of the participant.
* Language: Spoken language of the participant ordered by proficiency.
* ExperienceWithVR: Experience with VR (1: No experience; 5: Heavy user).
* AcquaintanceWithInterlocutor: Acquaintance with the interlocutor (1: Well-known; 5: Meeting for the first time).
* ExpTime: Merged Date and Time.
* PlayerID: ID of the participant related to the tracking data.
* ServerTime: First login time of the participant and with that tracking start time.
* ExperimentStart: Participant Login.
* ExperimentEnd: Participant Logout.
* DialogStart: Seconds after ExperimentStart (first logged in user) when the dialogue started.
* DialogEnd: Seconds after ExperimentStart (first logged in user) when the dialogue ended. 
* ExpLang: Spoken language during the experiment. Default is German.


## tracking_json_zip
For every participant, there is a zipped JSON file containing the tracking data.

## crisper_whisper_json
For every participant, the transcription of the dialogue is provided in a JSON file.
For data protection reasons, the audio files are currently not provided.
But we are testing the possibilities to anonymize the audio files to make them available in the future or on request.

## scripts
Python scripts to process the data. Will be more structured in the future.



# BibTeX

```bibtex
@inproceedings{Luecking:Voll:Rott:Henlein:Mehler:2025-fraga,
  title     = {Head and Hand Movements During Turn Transitions: Data-Based Multimodal
               Analysis Using the {Frankfurt VR Gesture--Speech Alignment Corpus}
               ({FraGA})},
  author    = {Lücking, Andy and Voll, Felix and Rott, Daniel and Henlein, Alexander
               and Mehler, Alexander},
  year      = {2025},
  booktitle = {Proceedings of the 29th Workshop on The Semantics and Pragmatics
               of Dialogue -- Full Papers},
  series    = {SemDial'25 -- Bialogue},
  publisher = {SEMDIAL},
  url       = {http://semdial.org/anthology/Z25-Luecking_semdial_3316.pdf},
  pages     = {146--156}
}
```