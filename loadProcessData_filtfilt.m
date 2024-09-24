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
% Directory
% Write down where converted data file downloaded (file directory)
%dd='file path\';
%dd='/home/michael/Documents/Aston/MultimodalFW/Jeong11tasks/Data/';
%cd 'file path';
%cd '/home/michael/Documents/Aston/MultimodalFW/Jeong11tasks/Data/';
% Example: dd='Downlad_folder\SampleData\plotScalp\';
%addpath(genpath('/home/michael/Documents/Aston/MultimodalFW/Jeong11tasks/Scripts/Reference_toolbox/bbci_toolbox/'));

dd='H:\Jeong11tasks_data\m_files\';
cd 'H:\Jeong11tasks_data\m_files\';
addpath(genpath('C:\Users\pritcham\Documents\Jeong11tasks\Scripts\Scripts\Reference_toolbox\bbci_toolbox\'))

csv_dir='H:\Jeong11tasks_data\FiltFilt_EEG\';

datedir = dir('*.mat');
filelist = {datedir.name};

% Setting time duration: interval 0~3 s
ival=[0 3001];

%% Performance measurement
for i = 1:length(filelist)
    filelist{i}
    [cnt,mrk,mnt]=eegfile_loadMatlab([dd filelist{i}]);
    
    % Band pass filtering, order of 4, range of [8-30] Hz (mu-, beta-bands)
    filterBank = {[2 30]}; % incluidng lower frequencies
    for filt = 1:length(filterBank)
        clear epo_check epo epoRest
        filelist{i}
        filterBank{filt}
        
        cnt = proc_selectChannels(cnt, {'FC5','FC3','FC1','FC2','FC4','FC6',...
            'C5','C3','C1', 'Cz', 'C2', 'C4', 'C6',...
            'CP5','CP3','CP1','CPz','CP2','CP4','CP6'});
        
        
        %addpath '/home/michael/Documents/Aston/MultimodalFW/Jeong11tasks/Scripts'
        addpath 'C:\Users\pritcham\Documents\Jeong11tasks\Scripts\Scripts'
        
        cnt_FF = proc_filtfilt_Butter(cnt, 4, filterBank{filt});
        epo = cntToEpo(cnt_FF, mrk, ival);
              
        
       
        classes=size(epo.className,2);
        
        trial=50;
        
        %{
        trial=50;
        % Set the number of rest trial to the same as the number of other classes trial.
        for ii =1:classes
            if strcmp(epo.className{ii},'Rest')
                epoRest=proc_selectClasses(epo,{epo.className{ii}});
                epoRest.x=datasample(epoRest.x,trial,3,'Replace',false);
                epoRest.y=datasample(epoRest.y,trial,2,'Replace',false);
            else
                epo_check(ii)=proc_selectClasses(epo,{epo.className{ii}});
                
                % Randomization
                epo_check(ii).x=datasample(epo_check(ii).x,trial,3,'Replace',false);
                epo_check(ii).y=datasample(epo_check(ii).y,trial,2,'Replace',false);
            end
        end
        if classes<4
            epo_check(size(epo_check,2)+1)=epoRest;
        end
        %}

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
        %if classes<4
        if ~any(strcmp([epo_check(:).className],'Rest'))
            epo_check(size(epo_check,2)+1)=epoRest;
        end
        
        % concatenate the classes %concat not needed if no CSP
        %{
        for ii=1:size(epo_check,2)
            if ii==1
                concatEpo=epo_check(ii);
            else
                concatEpo=proc_appendEpochs(concatEpo,epo_check(ii));
            end
        end
        
        %}


        done_slicing=false;
        if done_slicing==false
            for j=1:length(epo_check)
                epoclass=epo_check(j);
                gesture=epoclass.className;
                % 50 trials per 3 grasps + rest per 3 sessions = 600 total
                filename=split(epoclass.file,'\');
                filename=filename{end};
                fnameparts=split(filename,'_');
                subject=strcat('00',erase(fnameparts{3},'sub'));
                session=erase(fnameparts{2},'session');
                for jj = 1:trial
                    classTable=array2table([epoclass.t',epoclass.x(:,:,jj)],'VariableNames',[{'Timestamp'},epoclass.clab]);
                    csvname=string(strcat(subject,'_',session,'-',gesture,'-',int2str(jj),'.csv'));
                    csvpath=strcat(csv_dir,csvname);
                    writetable(classTable,csvpath);
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

