# 13 Dec 2023
## To do:
- Create a system for systematically running different models with different data to compare results. DONE
- Change main.py to train.py and establish it as a module which can be used to train from different inputs DONE
- Create LUT for features to feature name to aid with training from different models DONE
(Later)
- Fine tune most effective models using hyperparameters
- Improve metrics reporting to include model hyper parameters
- If error is encountered, the CSV should feature error, then move on to next. (suggestion)

## Structure for model comparison:
- Main script controls which features and models should be used DONE
- Load and compile data using compile_data.py module DONE
- Train using this data with first model DONE
- Save metrics to a csv DONE
- Continue training models with the same data until all models in the list have been trained and metrics saved. DONE
- Move to next feature set and repeat DONE
- Script completes when all feature sets and models have been used to save data DONE

# 14 Dec 2023
## Issues:
- Compile_data is not working for IDs that are less than 10000, since it is trying to find ID 1, but the file ID is 00001
- Not using 2nd and 3rd frames in relevent features

# 20 Dec 2023
## To do:
- Hyperparameter tuning setup & run
- Sort out 2nd and 3rd frames
- Modify caching system to identify incomplete cache files based on:
    - Number of rows?
    - Size of file?
    - Any better way? What if rows are not always the same?

## Second and third frame approach/considerations:
- Change caching sytem to be compatible with 3 different sets of frames
- Change feature system to find features with frames and treat them as different feature sets for training and evaluation.
- Identify features with multiple frames:
    - If we know the number of videos, we should be able to divide the total number of files like this to find number of frames
    - File format indicates presence of frames
    - I can dictate which features have frames
- Check how the models take features - could they have all the frames and dimensions dictated?

# 10 Jan 2024
## Notes:
- Ran code but wasn't working, issue with y_train values containing NaN values.
- Identified issue related to my compile_data module, I never finished updating it to take 3 frames.
- I believe I have fixed the issue. due to handling of ID values the lookup was no longer working.
- Running the tests with baysian ridge provided similar results to last successful results
- next step, tune hyper parameters, then use this for depression/alexithymia
- Also planning to move logging config to new module.

# 11 Jan 2024 - *Memorability projected duplicated to start depression severity work.*
## Notes:
- 


# 21 Jan 2024 - *F0 Extraction Script.*
## Notes:
- script working to extract F0
- Only working with C7 note, so may be missing some info.
- Leading zeros shows need to remove silent spaces from start and end (potentially).
- Next, work on extracting resnet50 from dataset and further preprocessing.
- Extract resnet created with chatgpt but not yet tested.

# 22 Jan 2024 - *ResNet50 extraction.*
## Notes:
- Extracted ResNet50 and completed for 1400 files.
- first extraction is shape 1, 7, 7, 2048
- Looked at paper: 
    - https://d1wqtxts1xzle7.cloudfront.net/98893301/Smart_voice_recognition_based_on_deep_learning_for_depression_diagnosis-libre.pdf?1676887568=&response-content-disposition=inline%3B+filename%3DSmart_voice_recognition_based_on_deep_le.pdf&Expires=1705926501&Signature=J2iRYZkuDlXanKaxetny3wh0ZE6LVOmZz-4QAmiagzOf4K6yw6O~Du~gEvr9~h5bLHC~10JmK59OphBO09Tx~xKx1vFyipkXB1GLSVhV8gOcpMPiv~IYwTJD3468AwuDYM2m3uLCzAgpjUhoovizNoM5muVo2dGBrjFFshVE3KYkd24PHUrd6NIl2FDhExh8SgcTwuirjlfwgnEaXlK~jgTY7~TmS5MZ-cAoPp44K8hjwZ08-RXMVGQpjcZLRAS9TldxgZ28hSvFHOEA0QQj~q59YEU~YxdfEj6ttOtUXskh-pJsht-iPXdhL~AbVlfce~KvpnaFtTlXlbhflrDHSQ__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA
    - Important to use blackman non-dct window algorithm.

# 24 Jan 2024 - *Unet visualisation
## Notes
- Working on unet
- have some code that might work, facing issues with data compile
- have solved obvious issues, but at some point in compile module the dataframe expects ~4000 columns buts gets a fuck load.
- Working on fix, pick back up by running code and examining traceback error.
- Suspect could be an issue with dimensionality.

# 25 Jan 2024
## Notes
- Trying to fix f0 issue where data is not compiled due to different shapes 
- attempted to fix issue using a max feature column variable and padding
- Issues caused by one file much bigger than the rest:
    - 60ce3582c299649c634dca1f.npy
    - This file has over 70,000 columns compared to the rest's sub 10,000.
    - All values in this file are 0.
    - Need to address issue in f0 extraction for files like this.
    - for now will just delete.