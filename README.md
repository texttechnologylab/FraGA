[![Paper](http://img.shields.io/badge/paper-SemDial-B31B1B.svg)](https://www.semdial.org/anthology/Z25-Luecking_semdial_3316.pdf)
[![Conference](http://img.shields.io/badge/conference-SemDial--2025-4b44ce.svg)](https://semdial2025.github.io/)
[![version](https://img.shields.io/github/license/texttechnologylab/FraGA)]()

# FraGA

Head and hand movements during turn transitions: data-based multimodal analysis using the Frankfurt VR Gesture–Speech
Alignment Corpus

# Abstract

We introduce FraGA, a VR-based corpus of direction giving dialogues following the model of
the [SaGA](https://www.phonetik.uni-muenchen.de/Bas/BasSaGAdeu.html) corpus.
The tracking data of FraGA are used to carry out multimodal computing:
we look at turn transitions and re-evaluate findings from the literature on realworld dialogues and compare them with
avatar-mediated VR dialogues.
Interestingly, the established temporal overlap patterns of turns could not be replicated,
and no significant amount of partner-directed head orientation (approximating gaze) was observed around turn transition
points.
However, the special status of hesitations is evidenced by the co-occurring head movements, but not by hand movements.
In addition, we apply pink noise distribution fitting to the dialogue data, in particular to the hand movements.
Here we find that hand movements indeed follow 1⁄f fluctuations, a property of “meta-stable” dynamic systems.

# Data

## FraGa_allPlayer_playerID.xlsx

* Date: Date of the experiment.
* Time: Schedule time of the experiment.
* Person: Role of the participant (Person 1: Router; Person 2: Follower).
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

* body.json: Body tracking data.
* eye.json: Eye tracking data.
* facial.json: Facial tracking data.
* fingersleft.json: Left hand finger tracking data.
* fingersright.json: Right hand finger tracking data.
* handleft.json: Left hand tracking data.
* handright.json: Right hand tracking data.
* head.json: Head tracking data.
* misc.json: Selected Avatar IDs.
* objects.json: Menu and Object interaction data (grasped, ungrasped).

### facial.json

```C#
        Brow_Lowerer_L = 0,
        Brow_Lowerer_R = 1,
        Cheek_Puff_L = 2,
        Cheek_Puff_R = 3,
        Cheek_Raiser_L = 4,
        Cheek_Raiser_R = 5,
        Cheek_Suck_L = 6,
        Cheek_Suck_R = 7,
        Chin_Raiser_B = 8,
        Chin_Raiser_T = 9,
        Dimpler_L = 10,
        Dimpler_R = 11,
        Eyes_Closed_L = 12,
        Eyes_Closed_R = 13,
        Eyes_Look_Down_L = 14,
        Eyes_Look_Down_R = 15,
        Eyes_Look_Left_L = 16,
        Eyes_Look_Left_R = 17,
        Eyes_Look_Right_L = 18,
        Eyes_Look_Right_R = 19,
        Eyes_Look_Up_L = 20,
        Eyes_Look_Up_R = 21,
        Inner_Brow_Raiser_L = 22,
        Inner_Brow_Raiser_R = 23,
        Jaw_Drop = 24,
        Jaw_Sideways_Left = 25,
        Jaw_Sideways_Right = 26,
        Jaw_Thrust = 27,
        Lid_Tightener_L = 28,
        Lid_Tightener_R = 29,
        Lip_Corner_Depressor_L = 30,
        Lip_Corner_Depressor_R = 31,
        Lip_Corner_Puller_L = 32,
        Lip_Corner_Puller_R = 33,
        Lip_Funneler_LB = 34,
        Lip_Funneler_LT = 35,
        Lip_Funneler_RB = 36,
        Lip_Funneler_RT = 37,
        Lip_Pressor_L = 38,
        Lip_Pressor_R = 39,
        Lip_Pucker_L = 40,
        Lip_Pucker_R = 41,
        Lip_Stretcher_L = 42,
        Lip_Stretcher_R = 43,
        Lip_Suck_LB = 44,
        Lip_Suck_LT = 45,
        Lip_Suck_RB = 46,
        Lip_Suck_RT = 47,
        Lip_Tightener_L = 48,
        Lip_Tightener_R = 49,
        Lips_Toward = 50,
        Lower_Lip_Depressor_L = 51,
        Lower_Lip_Depressor_R = 52,
        Mouth_Left = 53,
        Mouth_Right = 54,
        Nose_Wrinkler_L = 55,
        Nose_Wrinkler_R = 56,
        Outer_Brow_Raiser_L = 57,
        Outer_Brow_Raiser_R = 58,
        Upper_Lid_Raiser_L = 59,
        Upper_Lid_Raiser_R = 60,
        Upper_Lip_Raiser_L = 61,
        Upper_Lip_Raiser_R = 62,
```

### fingersleft.json & fingersright.json

```C#
        Hand_WristRoot = 0, // root frame of the hand, where the wrist is located
        Hand_ForearmStub = 1, // frame for user's forearm
        Hand_Thumb0 = 2, // thumb trapezium bone
        Hand_Thumb1 = 3, // thumb metacarpal bone
        Hand_Thumb2 = 4, // thumb proximal phalange bone
        Hand_Thumb3 = 5, // thumb distal phalange bone
        Hand_Index1 = 6, // index proximal phalange bone
        Hand_Index2 = 7, // index intermediate phalange bone
        Hand_Index3 = 8, // index distal phalange bone
        Hand_Middle1 = 9, // middle proximal phalange bone
        Hand_Middle2 = 10, // middle intermediate phalange bone
        Hand_Middle3 = 11, // middle distal phalange bone
        Hand_Ring1 = 12, // ring proximal phalange bone
        Hand_Ring2 = 13, // ring intermediate phalange bone
        Hand_Ring3 = 14, // ring distal phalange bone
        Hand_Pinky0 = 15, // pinky metacarpal bone
        Hand_Pinky1 = 16, // pinky proximal phalange bone
        Hand_Pinky2 = 17, // pinky intermediate phalange bone
        Hand_Pinky3 = 18, // pinky distal phalange bone
```

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