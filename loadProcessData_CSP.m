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
clc; close all;% clear all;
clearvars -except remaining;
%%
% Directory
% Write down where converted data file downloaded (file directory)
dd='C:\Users\pritcham\Documents\Jeong11tasks\Data\'; 
cd 'C:\Users\pritcham\Documents\Jeong11tasks\Data\';
dd='H:\Jeong11tasks_data\m_files\';
cd 'H:\Jeong11tasks_data\m_files\';
% Example: dd='Downlad_folder\SampleData\plotScalp\';

csv_dir='C:\Users\pritcham\Documents\Jeong11tasks\SyncedRawCSVs\';
csp_csv_dir='H:\Jeong11tasks_data\Synced_CSP_CSVs\';

%mathworks.com/matlabcentral/answers/436023-add-toolbox-to-matlab-manually
addpath(genpath('C:\Users\pritcham\Documents\Jeong11tasks\Scripts\Scripts\Reference_toolbox\bbci_toolbox\'))

datedir = dir('*.mat');
filelist = {datedir.name};

%REMAINING LATE DOWNLOADS:
%Session 3 Sub 3, 5, 6, 7, 8, 9
%cellarr "remaining" has been made to manually hold these & be used instead
%of filelist on a second run

% Setting time duration: interval 0~3 s
ival=[0 3001];

%% Performance measurement
for i = 1:length(filelist)
    filelist{i}
    [cnt,mrk,mnt]=eegfile_loadMatlab([dd filelist{i}]);
    
    % Band pass filtering, order of 4, range of [8-30] Hz (mu-, beta-bands)
    filterBank = {[8 30]};
    for filt = 1:length(filterBank)
        clear epo_check csp_fv epo
        filelist{i};
        filterBank{filt}
    %what EXACTLY is filt doing? Line 126 we assign result per value of
    %filt then later take the highest result
    %line 51 we appear to be applying bworth filters according to filt?
    %WAIT IS THIS MAYBE SPLITTING INTO 1HZ FREQUENCY BANDS BTWN 8 - 30
    %{OHHH MY GOD OK SO FILTERBANK IS A  BANK  OF  FILTERS  LIKE YOU COULD
    %HAVE AND 8-30Hz AND ALSO A 5-20Hz AND A 1-100Hz AND  THAT  IS WHAT IT
    %WOULD ITERATE OVER NOT THE FILT FREQS THEMSELVES.
        
        cnt = proc_filtButter(cnt, 4 ,filterBank{filt});
        epo=cntToEpo(cnt,mrk,ival); %continuous to epochs, where epoch is marker+ival[1] to marker+ival[2]
        
        % Select channels
        epo = proc_selectChannels(epo, {'FC5','FC3','FC1','FC2','FC4','FC6',...
            'C5','C3','C1', 'Cz', 'C2', 'C4', 'C6',...
            'CP5','CP3','CP1','CPz','CP2','CP4','CP6'});
        
        classes=size(epo.className,2);
        
        trial=50;
        
        % Set the number of rest trial to the same as the number of other classes trial.
        for ii =1:classes
            if strcmp(epo.className{ii},'Rest')
                %epoRest=proc_selectClasses(epo,{epo.className{ii}});
                epoRest=proc_selectClasses(epo,epo.className(ii)); %supposedly faster

                if ~(size(epoRest.x,3)==trial)  %randomness dependent on subject/session for sync purposes
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
                %epo_check(ii)=proc_selectClasses(epo,{epo.className{ii}});
                epo_check(ii)=proc_selectClasses(epo,epo.className(ii));

                if ~(size(epo_check(ii).x,3)==trial)
                    disp('Need to downsample')

                    fnameparts=split(filelist{i},'_');
                    subject=strcat('00',erase(fnameparts{3},'sub'));
                    session=erase(fnameparts{2},'session');
                    unique=str2double(strcat(subject,session));
                    stream=RandStream('mt19937ar','Seed',unique);
                
                     % Randomization
                    epo_check(ii).x=datasample(stream,epo_check(ii).x,trial,3,'Replace',false); %selects 50 of each class
                    epo_check(ii).y=datasample(stream,epo_check(ii).y,trial,2,'Replace',false);
                end
            end
        end
        if classes<4
            epo_check(size(epo_check,2)+1)=epoRest;
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        done_non_neutrals=true;
        done_neutrals=true;

        if done_non_neutrals==false
            for j=1:length(epo_check)
                epoclass=epo_check(j);
                gesture=epoclass.className;
                % 50 trials per 3 grasps per 3 sessions = 450 total
                filename=split(epoclass.file,'\');
                filename=filename{end};
                fnameparts=split(filename,'_');
                subject=strcat('00',erase(fnameparts{3},'sub'));
                session=erase(fnameparts{2},'session');
                for jj = 1:trial
                    classTable=array2table([epoclass.t',epoclass.x(:,:,jj)],'VariableNames',[{'Timestamp'},epoclass.clab]);
                    csvname=string(strcat(subject,'_',session,'-',gesture,'-',int2str(jj),'.csv'));
                    csvpath=strcat(csv_dir,csvname);
                    %writetable(classTable,csvpath);
                    %print('write this to a CSV now')
                end
            end
        end

        if done_neutrals==false
            gesture=epoRest.className;
            % 50 trials per 3 grasps per 3 sessions = 450 total
            filename=split(epoRest.file,'\');
            filename=filename{end};
            fnameparts=split(filename,'_');
            subject=strcat('00',erase(fnameparts{3},'sub'));
            session=erase(fnameparts{2},'session');
            for jj = 1:trial
                classTable=array2table([epoRest.t',epoRest.x(:,:,jj)],'VariableNames',[{'Timestamp'},epoRest.clab]);
                csvname=string(strcat(subject,'_',session,'-',gesture,'-',int2str(jj),'.csv'));
                csvpath=strcat(csv_dir,csvname);
                %writetable(classTable,csvpath);
                %print('write this to a CSV now')
            end
        end

        lum1array=[epo_check(3).t',epo_check(3).x(:,:,1)];
        lum1Table=array2table(lum1array,'VariableNames',[{'Timestamp'},epo_check(3).clab]);

%         continue

      %%% appending Rest to epo_check
        epo_check(size(epo_check,2)+1)=epoRest;

        
        % concatenate the classes
        for ii=1:size(epo_check,2)
            if ii==1
                concatEpo=epo_check(ii);
            else
                concatEpo=proc_appendEpochs(concatEpo,epo_check(ii));
            end
        end
     
        %% CSP - FEATURE EXTRACTION
        [csp_fv,csp_w,csp_eig]=proc_multicsp(concatEpo,3);
        %"we calculated a transformatoiun matrix using CSP in which the log
        %variances of the firat and last 3 columns (maybe what the 3
        %argument is?) were used as a feature"


     %%% put the CSP into a CSV (lol)
        filename=split(csp_fv.file,'\');
        filename=filename{end};
        fnameparts=split(filename,'_');
        subject=strcat('00',erase(fnameparts{3},'sub'));
        session=erase(fnameparts{2},'session');
        for k=1:size(concatEpo.y,2)     %loop through each trial (each has a single y)
            gesture=csp_fv.className(ceil(k/trial));

            cspTable=array2table([csp_fv.t',csp_fv.x(:,:,k)],'VariableNames',[{'Timestamp'},strcat('CSP_',csp_fv.clab)]);
            %rep=mod(k,trial); %not quite, mode(50,50) is 0 but i want 50
            rep=k-trial*(ceil(k/trial)-1);
            csvname=string(strcat(subject,'_',session,'-',gesture,'-',int2str(rep),'.csv'));
            csvpath=strcat(csp_csv_dir,csvname);
            writetable(cspTable,csvpath);
            %print('write this to a CSV now')
        end
%{
        proc=struct('memo','csp_w');
        
        proc.train= ['[fv,csp_w]=  proc_multicsp(fv, 3); ' ...
            'fv= proc_variance(fv); ' ...
            'fv= proc_logarithm(fv);'];
        
        proc.apply= ['fv= proc_linearDerivation(fv, csp_w); ','fv= proc_variance(fv); ' ,'fv= proc_logarithm(fv);'];
%}    

        % they classify with random 10fold across ALL CLASSES ALL SESSIONS
        % they classify over ONLY 3 GESTURES (no rest)
        % they reach about 0.4 accuracy in most ppts
    %{    
        %% RLDA - CLASSIFICATION WITH 10-FOLD CROSS-VALIDATION       
        [C_eeg, loss_eeg_std, out_eeg.out, memo] = xvalidation(concatEpo,'RLDAshrink','proc',proc, 'kfold', 10);
        Result(filt)= 1-C_eeg;
        Result_Std(filt)=loss_eeg_std;
        All_csp_w(:,:,filt)=csp_w;
  %}
    end   
    % Maximum classification performance of each subject
    %maxPerformance(i) = max(Result);
    
end
%{
A = num2cell(maxPerformance);
subPerformance = cat(1, filelist, A);

% Save results of FBCSP with RLDA in excel file
% total results: 9 bands of accuracies
filename = 'Performance_reaching_ME.xlsx';
writecell((subPerformance)', filename, 'Sheet', 1);
%}
