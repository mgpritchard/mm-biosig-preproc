% -------------------------------------------------------------------------
% ACC_HandGrasping.m
% This source code is about to measure the classification accuracy based on the basic machine learning method.
% This classification accuracy was computed by usign the fundamental EEG signal processing such band-pass filtering, time epoch, feature extraction, classification.
% Feature extraction: Common spatial pattern (CSP),
% Classifier: regularized linear discriminant analysis (RLDA)

% If you want to improve the classification accuracies, you could adopt more advanced methods. 
% For example: 
% 1. Independent component analysis(ICA) for artifact rejection
% 2. Common average reference (CAR) filter, Laplacian spatial filter, and band power for feature extraction
% 3. Support vector machine (SVM) or Random forest (RF) for classifier 

% Please add the bbci toolbox in the 'Reference_toolbox' folder
%--------------------------------------------------------------------------

%% Initalization
clc; close all; clear all;
%%
addpath(genpath('C:\Users\pritcham\Documents\Jeong11tasks\Scripts\Scripts\Reference_toolbox\bbci_toolbox\'))
% Directory
% Write down where converted data file downloaded (file directory)
dd='H:\Jeong11tasks_data\EMG\EMG_ConvertedData\'; 
cd 'H:\Jeong11tasks_data\EMG\EMG_ConvertedData\';
% Example: dd='Downlad_folder\SampleData\plotScalp\';

%mathworks.com/matlabcentral/answers/436023-add-toolbox-to-matlab-manually
%addpath(genpath())

csv_dir='H:\Jeong11tasks_data\EMG\Raw_EMGCSVs\';
%csv_dir='H:\Jeong11tasks_data\RawEMG_syncedto_RawEEG\'; %it will already
%be synced because the sync is about downsampling and this is based on
%unique combo of pptID and recording session

datedir = dir('*.mat');
filelist = {datedir.name};
idx=contains(filelist,'grasp')&contains(filelist,'real');
filelist=filelist(idx);

% Setting time duration: interval 0~3 s
ival=[0 3001];

%% Performance measurement
for i = 1:length(filelist)
    filelist{i}
    [cnt,mrk,mnt]=eegfile_loadMatlab([dd filelist{i}]);
    
    % Band pass filtering, order of 5, range of [10-500] Hz (per Jeong)
    filterBank = {[10 500]};
    for filt = 1:length(filterBank)
        clear epo_check epo epoRest
        filelist{i}
        filterBank{filt}
        
        cnt = proc_filtButter(cnt, 5 ,filterBank{filt});
        epo=cntToEpo(cnt,mrk,ival);
        epo.x=abs(epo.x); %rectifying EMG
        % Select channels 
        
        epo = proc_selectChannels(epo, {'EMG_1','EMG_2','EMG_3','EMG_4',...
            'EMG_5','EMG_6'}); %all except EMG_ref which is elbow

        %for ref i THINK 6=bicep, 5=tricep, 4=flexcarp uln, 3=flexcarp rad,
        % 2 = ext carp uln, 1 = ext digitorum
        % each of those might be swapped 1-2, 3-4, 5-6 though        

        classes=size(epo.className,2);
        
        trial=50;
        
        % Set the number of rest trial to the same as the number of other classes trial.
        for ii =1:classes
            if strcmp(epo.className{ii},'Rest')
                epoRest=proc_selectClasses(epo,{epo.className{ii}});


                if ~(size(epoRest.x,3)==trial)
                    disp('Need to downsample')

                    fnameparts=split(filelist{i},'_');
                    subject=strcat('00',erase(fnameparts{3},'sub'));
                    session=erase(fnameparts{2},'session');
                    unique=str2double(strcat(subject,session));
                    stream=RandStream('mt19937ar','Seed',unique);
                    

                    epoRest.x=datasample(stream,epoRest.x,trial,3,'Replace',false);
                    epoRest.y=datasample(stream,epoRest.y,trial,2,'Replace',false);
                end
            else
                epo_check(ii)=proc_selectClasses(epo,{epo.className{ii}});
                
                if ~(size(epo_check(ii).x,3)==trial)
                    disp('Need to downsample')
                    % Randomization
                    % this takes a random [trial] number of epochs from
                    % epo_check (Replace false means no replacement ie no item
                    % can be selected more than once). In our case there are
                    % already only [trial] epochs so this just shuffles them.
                    % In a processed epo_check(n), y should be consistent
                    % across n as we have just selected n to be the epo_check
                    % for a given class. Therefore shuffling y will have no
                    % impact and is ok to do separately from x. (We could get
                    % indices from x a& use those to get y if it was an issue).

                    fnameparts=split(filelist{i},'_');
                    subject=strcat('00',erase(fnameparts{3},'sub'));
                    session=erase(fnameparts{2},'session');
                    unique=str2double(strcat(subject,session));
                    stream=RandStream('mt19937ar','Seed',unique);

                    epo_check(ii).x=datasample(stream,epo_check(ii).x,trial,3,'Replace',false);
                    epo_check(ii).y=datasample(stream,epo_check(ii).y,trial,2,'Replace',false);
                end
            end
        end
        %if classes<4
        if ~any(strcmp([epo_check(:).className],'Rest'))
            epo_check(size(epo_check,2)+1)=epoRest;
        end
        
        % concatenate the classes
        for ii=1:size(epo_check,2)
            if ii==1
                concatEpo=epo_check(ii);
            else
                concatEpo=proc_appendEpochs(concatEpo,epo_check(ii));
            end
        end

        done_slicing=false;
        if done_slicing==false
            for j=1:length(epo_check)
                epoclass=epo_check(j);
                gesture=epoclass.className;
                % 50 trials per 3 grasps per 3 sessions = 450 total
                filename=split(epoclass.file,'\');
                filename=filename{end};
                fnameparts=split(filename,'_');
                subject=strcat('00',erase(fnameparts{3},'sub'));
                session=erase(fnameparts{2},'session');
                if isequal(gesture{1},'Rest') %TEMP TO MOP UP RESTS
                    for jj = 1:trial
                        classTable=array2table([epoclass.t',epoclass.x(:,:,jj)],'VariableNames',[{'Timestamp'},epoclass.clab]);
                        csvname=string(strcat(subject,'_',session,'-',gesture,'-',int2str(jj),'.csv'));
                        csvpath=strcat(csv_dir,csvname);
                        writetable(classTable,csvpath);
                     %print('write this to a CSV now')
                    end
                end
            end
        end





       %{ 
        %% CSP - FEATURE EXTRACTION
        [csp_fv,csp_w,csp_eig]=proc_multicsp(concatEpo,3);
        proc=struct('memo','csp_w');
        
        proc.train= ['[fv,csp_w]=  proc_multicsp(fv, 3); ' ...
            'fv= proc_variance(fv); ' ...
            'fv= proc_logarithm(fv);'];
        
        proc.apply= ['fv= proc_linearDerivation(fv, csp_w); ','fv= proc_variance(fv); ' ,'fv= proc_logarithm(fv);'];
       %}
        %{
        %% RLDA - CLASSIFICATION WITH 10-FOLD CROSS-VALIDATION       
        [C_eeg, loss_eeg_std, out_eeg.out, memo] = xvalidation(concatEpo,'RLDAshrink','proc',proc, 'kfold', 10);
        Result(filt)= 1-C_eeg;
        Result_Std(filt)=loss_eeg_std;
        All_csp_w(:,:,filt)=csp_w;
        %}
    end   
    % Maximum classification performance of each subject
   % maxPerformance(i) = max(Result);
    
end
%{
A = num2cell(maxPerformance);
subPerformance = cat(1, filelist, A);

% Save results of FBCSP with RLDA in excel file
% total results: 9 bands of accuracies
filename = 'Performance_reaching_ME.xlsx';
writecell((subPerformance)', filename, 'Sheet', 1);
%}

