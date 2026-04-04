 # added by Anita Rau April 2025

from vlmeval.smp import *
from sharedeval import precision, jaccard, recall, f1, map_for_classification, mloc_iou, f1max, f1max_thres, accuracy
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sharedeval.data.cholec_helpers import *

import torch
import os
from difflib import get_close_matches 
device = "cuda" if torch.cuda.is_available() else "cpu"


def infer_data(
    model, 
    work_dir, 
    name, 
    dataset, 
    task, 
    prompt, 
    **kwargs
):
    """
    Predicts and evaluates data, while eval_data only evaluates
    """
    # Different models have different attributes:
    if isinstance(model.model, str):
        model_name = model.model
    elif hasattr(model, 'model_path'):
        model_name = model.model_path.split('/')[1]
    else:
        model_name = model.name
    if 'paligemma' in model_name:
        # paligemma needs its own inference
        return infer_data_paligemma(model, work_dir, name, dataset, task, prompt, **kwargs)
    write_dir = osp.join(work_dir, task['name'], model_name, name)
    if not osp.exists(write_dir):
        os.makedirs(write_dir)

    # save prompt as txt
    with open(osp.join(write_dir, 'prompt.txt'), 'w') as f:
        if isinstance(prompt, list):
            f.write(','.join(prompt)) # for few shot
        else:
            f.write(prompt)

    # choose zero-shot or few-shot prompt
    if 'fewshot' in task['name']:
        def eval_model(frame, prompt):
            return model.generate(prompt + [frame, 'output: '] )
    else:
        def eval_model(frame, prompt):
            return model.generate([frame, prompt])

    # as API calls often fail, we save all predictions as json files, and then read those in an eval loop
    for frame, label in tqdm(dataset):
        out_file = osp.join(write_dir, f'{"-".join(frame["path"].split("/")[-3:])}.json')
        if osp.exists(out_file) and not kwargs['override_outputs']: #TODO double check this
            continue
        else:
            if osp.exists(out_file):
                os.remove(out_file)
            try:
                ret = eval_model(frame['path'], prompt) 
                if isinstance(ret, int):  # returns int if gemini blocks the request (e.g. image contains a lot of blood)
                    dump('Blocked: ' + str(ret), out_file)
                    continue
                else: 
                    start_index = ret.find('{')
                    end_index = ret.rfind('}')  # Use rfind to get the last occurrence of '}'

                    if start_index != -1 and end_index != -1:
                        result = ret[start_index:end_index + 1]  # Include the '}'
                    else:
                        result = "{}"  # Handle case where {} are not found # was None
                    # clean up result:
                    ret = result.strip("```").strip("json").replace("\n","").replace("False", "0").replace("True", "1").replace("false", "0").replace("true", "1")
                    pred = json.loads(ret)
                    dump(pred, out_file)

            except Exception as e:
                print('Exception: ' + str(e))
                dump('Exception: ' + str(e), out_file)
                continue

    preds, labels = eval_data(model, work_dir, name, dataset, task)
    return preds, labels


def infer_data_video(
    model, 
    work_dir, 
    name, 
    dataset, 
    task, 
    prompt, 
    nframe=8,  
    **kwargs
):
    """
    Predicts and evaluates data (video specific)
    """
    # Different models have different attributes:
    if isinstance(model.model, str):
        model_name = model.model
    elif hasattr(model, 'model_path'):
        model_name = model.model_path.split('/')[1]
    else:
        model_name = model.name
    write_dir = osp.join(work_dir, task['name'], model_name, name)
    if not osp.exists(write_dir):
        os.makedirs(write_dir)

    # save prompt as txt
    with open(osp.join(write_dir, 'prompt.txt'), 'w') as f:
        if isinstance(prompt, list):
            f.write(','.join(prompt))
        else:
            f.write(prompt)
    
    # keep track of dataset name
    dataset_name = dataset.dataset_name
    print(f'Video model: <{model_name}> for dataset: <{dataset_name}>')
    print('-'*60)

    # choose zero-shot or few-shot prompt
    if 'fewshot' in task['name']:
        def eval_model(video_path, prompt):
            raise NotImplementedError('Few-shot prompt not implemented for video data!')
    
    elif (
        'Phi-3.5-Vision' in model_name or \
        'InternVL2' in model_name or \
        'Qwen2-VL' in model_name or \
        'gpt-4o' in model_name
    ):  
        def eval_model(video_path, prompt):
            if 'gpt-4o' in model_name:
                nframes = 35
            elif 'Phi-3.5-Vision' in model_name:
                nframes = 35
            elif 'Qwen2-VL' in model_name:
                nframes = 35
            elif 'InternVL2' in model_name:
                nframes = 70
            prompt = prompt.replace('<NUM_SAMP>', str(nframes))
            message = [dict(type='text', value=prompt)]


            frames_presaved = 'jigsaws' in task['name'] or 'autolaparo' in task['name'] or 'heichole_skill_assessment' in task['name']
            if frames_presaved:
                # assume frames are already saved
                if 'jigsaws_skill_assessment' in task.name:
                    frame_dir = '/'.join(video_path['path'].split('/')[-3:]).replace('jigsaws_', '')
                    frame_dir = os.path.join(f'../data/jigsaws_skill_assessment', frame_dir)
                    frame_dir = frame_dir.replace('.mp4', '_images')
                else:
                    frame_dir = video_path.replace('.mp4', '_images')
                
                # NOTE: does not follow nframes, rather based on the saved presaved frame count
                num_frames = sum(1 for f in os.listdir(frame_dir) if os.path.isfile(os.path.join(frame_dir, f)))
                frame_idxs = list(range(num_frames))
                # can sample frames here if it does not fit in memory

                for frame_i in frame_idxs:
                    im = os.path.join(frame_dir, f'frame-{frame_i}.jpg')
                    message.append(dict(type='image', value=im))

            else:
                # write frames to a temp dir for inference
                tmp_frame_dir = osp.join(write_dir, osp.basename(video_path['path']))
                os.makedirs(tmp_frame_dir, exist_ok=True)
                
                # check if frames are already extracted
                if len(os.listdir(tmp_frame_dir)) != nframes:
                    print(
                        'WARNING: Re-extracting frames for video: ', video_path['path']
                    )
                    shutil.rmtree(tmp_frame_dir)
                    os.makedirs(tmp_frame_dir)
                    
                    video = decord.VideoReader(video_path['path'])
                    frame_indices = list(range(0, len(video), len(video) // nframes))[:nframes]  
                    frames = list(video.get_batch(frame_indices).asnumpy())

                    for frame_idx, frame in tqdm(zip(frame_indices, frames), total=nframes, desc='Saving frames'):
                        frame_path = osp.join(tmp_frame_dir, f'frame-{frame_idx}.jpg')
                        if not osp.exists(frame_path):
                            frame_pil = Image.fromarray(frame)
                            target_size = (854, 480)    # default is 480p 
                            frame_pil.thumbnail(target_size, Image.Resampling.LANCZOS)
                            frame_pil.save(frame_path)
                        message.append(dict(type='image', value=frame_path))
                else:
                    for frame in os.listdir(tmp_frame_dir):
                        message.append(dict(type='image', value=osp.join(tmp_frame_dir, frame)))
            return model.generate(message=message, dataset=dataset_name)
    
    else:
        def eval_model(video_path, prompt):
            message = [
                dict(type='text', value=prompt), 
                dict(type='video', value=video_path['path'])
            ]
            return model.generate(message=message, dataset=dataset_name)

    # as API calls often fail, we save all predictions as json files, and then read those in an eval loop
    for video_path, label in tqdm(dataset):
        out_file = osp.join(write_dir, f'{"-".join(video_path["path"].split("/")[-3:])}.json')
        if osp.exists(out_file) and not kwargs['override_outputs']:
            continue
        else:
            if osp.exists(out_file):
                os.remove(out_file)
            if True: #try:
                if 'error_detection' in task['name']:
                    prompt = prompt.replace('<ERROR_TYPE>', label['error_type'])
                ret = eval_model(video_path, prompt)
                if isinstance(ret, int):  # returns int if gemini blocks the request (e.g. image contains a lot of blood)
                    dump('Blocked: ' + str(ret), out_file)
                    continue
                else: 
                    # clean up result:
                    if 'jigsaws' in task['name'] or 'autolaparo' in task['name'] or 'heichole_skill_assessment' in task['name']:
                        # slightly different format based on prompt
                        pred = ret
                    else:
                        ret = ret.strip("```").strip("json").replace("\n","")
                        pred = json.loads(ret)
                    dump(pred, out_file)
            
            #except Exception as e:
            #    print('Exception: ' + str(e))
            #    dump('Exception: ' + str(e), out_file)
            #    continue
    print('-'*60)

    preds, labels = eval_data(model, work_dir, name, dataset, task)
    return preds, labels


def infer_data_paligemma(
    model, 
    work_dir, 
    name, 
    dataset, 
    task, 
    prompt, 
    **kwargs
):
    # needs on infer loop, as no json formatting promptable
    model_name = model.name
    write_dir = osp.join(work_dir, task['name'], model_name, name)
    if not osp.exists(write_dir):
        os.makedirs(write_dir)

    # save prompt as txt
    with open(osp.join(write_dir, 'prompt.txt'), 'w') as f:
        if isinstance(prompt, list):
            f.write(','.join(prompt))
        else:
            f.write(prompt)

    def eval_model(frame, prompt):
        return model.generate([frame, prompt])

    if "tool" in task['name'] or "cvs" in task['name'] or "heichole_action" in task['name'] or "dresden" in task['name']:
        target_length = len(dataset.labels[0][1])
        if "heichole_tool" in task['name']: #TODO needed?
            target_length = 7 # labels are padded with zeros
        elif "dresden" in task['name']:
            target_length = 11 # cut the 'null' class
        #    task.label_names = [i for i in task.label_names if not i =='null']
        for frame, label in tqdm(dataset):

            out_file = osp.join(write_dir, f'{"-".join(frame["path"].split("/")[-3:])}.json')
            if osp.exists(out_file) and not kwargs['override_outputs']:
                continue
            else:
                if osp.exists(out_file):
                    os.remove(out_file)

                outputs = dict()
                for i, p in enumerate(prompt):
                    ret = eval_model(frame['path'], p) 
                    if ret == 'no':
                        output = 0
                    elif ret == 'yes':
                        output = 1
                    else:
                        print(" ")
                        continue
                    outputs[task.label_names[i]] = output
                if len(outputs) != target_length:
                    print('Failed on', frame['path'])
                    continue
                dump(outputs, out_file)
    

    elif 'object_detection' in task['name']:
        def process_object_detection(output):
            import re
            loc_values = re.findall(r'<loc(\d{4})>', output)
            if len(loc_values) != 4:
                return None

            y0, x0, y1, x1 = map(int, loc_values)
            return y0, x0, y1, x1
        
        for frame, label in tqdm(dataset):
            out_file = osp.join(write_dir, f'{"-".join(frame["path"].split("/")[-3:])}.json')
            if osp.exists(out_file) and not kwargs['override_outputs']:
                continue
            else:
                if osp.exists(out_file):
                    os.remove(out_file)

                outputs = dict()
                for i, p in enumerate(prompt):
                    ret = eval_model(frame['path'], p)
                    num_found_objs = len(ret.split(';'))
                    for o, found_object in enumerate(ret.split(';')):
                        output = process_object_detection(found_object)
                        if output is None:
                            continue
                        if task.label_names[i] in ['tool', 'hand', 'forceps', 'needledriver', 'bovie']:  # endoscapes: only tool can have more than one instance. avos: all can be > 1
                            outputs[task.label_names[i]+str(o+1)] = list(output)
                        else:
                            outputs[task.label_names[i]] = list(output)
                dump(outputs, out_file)


    elif 'phase' in task['name']:
        if 'heichole' in task['name'] or 'cholec80' in task['name']:
            options = ['preparation', 'calot triangle dissection', 'clipping cutting', 'gallbladder dissection', 'gallbladder packaging', 'cleaning coagulation', 'gallbladder retraction']
        elif 'multibypass' in task['name']:
            options = ['preparation', 'gastric pouch creation', 'omentum division', 'gastrojejunal anastomosis', 'anastomosis test', 'jejunal separation', 'petersen space closure', 
                       'jejunojejunal anastomosis', 'mesenteric defect closure', 'cleaning & coagulation', 'disassembling', 'other intervention']
        else:
            raise NotImplementedError

        for frame, label in tqdm(dataset):
            out_file = osp.join(write_dir, f'{"-".join(frame["path"].split("/")[-3:])}.json')
            if osp.exists(out_file) and not kwargs['override_outputs']:
                continue
            else:
                if osp.exists(out_file):
                    os.remove(out_file)
                try:
                    outputs = dict()
                    ret = eval_model(frame['path'], prompt).strip().lower() 
                    closest_match = get_close_matches(ret, options, n=1)
                    outputs['phase'] = options.index(closest_match[0])
                    dump(outputs, out_file)

                except Exception as e:
                    dump('Exception: ' + str(e), out_file)
                    print('exception')

    elif 'avos_action' in task['name']:
        for frame, label in tqdm(dataset):
            out_file = osp.join(write_dir, f'{"-".join(frame["path"].split("/")[-3:])}.json')
            if osp.exists(out_file) and not kwargs['override_outputs']:
                continue
            else:
                if osp.exists(out_file):
                    os.remove(out_file)
                try:
                    outputs = dict()
                    ret = eval_model(frame['path'], prompt).strip().lower() 
                    options = ['cutting', 'tying knots', 'suturing', 'background task']
                    closest_match = get_close_matches(ret, options, n=1)
                    outputs['action'] = options.index(closest_match[0])
                    dump(outputs, out_file)

                except Exception as e:
                    dump('Exception: ' + str(e), out_file)
                    print('exception')
    elif "triplet" in task['name']:
        for frame, label in tqdm(dataset):
            out_file = osp.join(write_dir, f'{"-".join(frame["path"].split("/")[-3:])}.json')
            if osp.exists(out_file) and not kwargs['override_outputs']:
                continue
            else:
                if osp.exists(out_file):
                    os.remove(out_file)
                try:
                    outputs = {"instrument": [], "verb": [], "target": []}
                    ret = eval_model(frame['path'], 'Which instruments are present in this image? Choose all that apply from this list: grasper, bipolar, hook, scissors, clipper, irrigator, specimen bag, none')                   
                    if not 'one' in ret:
                        tools = ret.split(',')
                        if len(tools) > 1:
                            print('multiple tools detected')
                        for i, tool in enumerate(tools):
                            tool = tool.lower()
                            if not tool in ['grasper', 'bipolar', 'hook', 'scissors', 'clipper', 'irrigator', 'specimen bag']:
                                continue
                            #Return a  dict using this JSON schema: {"instrument": [tool1,...], "verb": [activity1,...], "target": [tissue1,...]}
                            p = f'What is the {tool} doing in this image? Choose one action from these options: grasp, retract, dissect, coagulate, clip, cut, aspirate, irrigate, pack, nothing.'
                            verb = eval_model(frame['path'], p)
                            if not verb in ['grasp', 'retract', 'dissect', 'coagulate', 'clip', 'cut', 'aspirate', 'irrigate', 'pack']:
                                verb = 'null'
                                subject = 'null'
                            else:
                                p = f'What is the {tool} {verb}ing? Choose one tissue from these options: gallbladder, cystic plate, cystic duct, cystic artery, cystic pedicle, blood vessel, fluid, abdominal wall cavity, liver, adhesion, omentum, peritoneum, gut, specimen bag, nothing.'
                                subject = eval_model(frame['path'], p)
                                if not subject in ['gallbladder', 'cystic plate', 'cystic duct', 'cystic artery', 'cystic pedicle', 'blood vessel', 'fluid', 'abdominal wall cavity', 'liver', 'adhesion', 'omentum', 'peritoneum', 'gut', 'specimen bag']:
                                    subject = 'null'
                            outputs["instrument"].append(tools[i])
                            outputs["verb"].append(verb)
                            outputs["target"].append(subject)

                    dump(outputs, out_file)

                except Exception as e:
                    dump('Exception: ' + str(e), out_file)
                    print('exception')
    else:
        raise NotImplementedError
    preds, labels = eval_data(model, work_dir, name, dataset, task)
    return preds, labels


def infer_data_contrastive(
    model, 
    work_dir, 
    name, 
    dataset, 
    task, 
    prompt, 
    **kwargs
):
    # Predict using CLIP, OpenCLIP and SurgVLP
    preds = []
    labels = []
    if 'cholec80' in task['name'] or 'heichole_tool' in task['name'] or 'heichole_phase' in task['name'] or 'cvs' in task['name'] or 'heichole_action' in task['name'] or 'multibypass' in task['name'] or 'dresden' in task['name'] or 'disease_severity' in task['name']:
        label_map = {idx: label for idx, label in enumerate(task['label_names'])}
    elif 'cholect45_triplet' in task['name']:
        triplet_file = osp.join(dataset.data_dir, 'dict/triplet.txt')
        label_map = {}
        with open(triplet_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(':')
                label_map[int(line[0])] = line[1]
    elif 'avos_action' in task['name']:
        label_map = {k: v for k, v in dataset.map.items()}
    else:
        raise NotImplementedError
    
    if isinstance(model.model, str):
        model_name = model.model
    elif isinstance(model.name, str):
        model_name = model.name
    else:
        model_name = model.model_path.strip('/')#VLM not API handled differently n VLMEvalKit

    for frame, label in tqdm(dataset):

        probabilities = model(prompt, frame['path'])
        if task['clip_eval_mode'] == 'singlelabel':
            pred = np.argmax(probabilities)
        elif task['clip_eval_mode'] == 'sigmoid':
            pred = probabilities[0] # remove extra dimension. Best threshold is chosen at evaluation time
        else:
            raise ValueError('Invalid eval_type')
        if task['name'] == 'cholect45_triplet_recognition':  # multimodal binary classification task, so null class is never positive, it would just be a zero vector
            pred = pred[:-1]  # remove null instrument
        elif task['name'] == 'heichole_tool_recognition':
            label = label[:len(pred)]  # GT is padded with zeros, so cut them.
        elif 'avos_action' in task['name']:
            label = label_map[label]
        labels.append(label)
        preds.append(pred)

    preds = np.array(preds)
    labels = np.array(labels)
    # write preds and labels to file
    write_dir = osp.join(work_dir, task['name'], model_name, name)
    if not osp.exists(write_dir):
        os.makedirs(write_dir)
    np.savez(osp.join(write_dir, 'results.npz'), labels=labels, preds=preds)

    eval_data_contrastive(model, work_dir, name, dataset, task, prompt, **kwargs)
    return preds, labels


def eval_data_contrastive(
    model, 
    work_dir, 
    name, 
    dataset, 
    task, 
    *args, 
    **kwargs
):
    # Eval for CLIP, OpenCLIP, and SurgVLP
    if isinstance(model.model, str):
        model_name = model.model
    elif isinstance(model.name, str):
        model_name = model.name
    else:
        model_name = model.model_path.strip('/')  # VLM not API handled differently n VLMEvalKit
    read_dir = osp.join(work_dir, task['name'], model_name, name)

    if 'cholec80' in task['name'] or 'heichole_tool' in task['name'] or 'heichole_phase' in task['name'] or 'cvs' in task['name'] or 'action' in task['name'] or 'multibypass' in task['name'] or 'dresden' in task['name'] or 'disease' in task['name']:
        label_map = {idx: label for idx, label in enumerate(task['label_names'])}
    elif 'cholect45_triplet' in task['name']:
        triplet_file = osp.join(dataset.data_dir, 'dict/triplet.txt')
        label_map = {}
        with open(triplet_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(':')
                label_map[int(line[0])] = line[1]
    else:
        raise NotImplementedError

    data = np.load(osp.join(read_dir, 'results.npz'))
    preds = data['preds']
    labels = data['labels']
    if 'dresden' in task['name']:
        labels = np.concatenate((labels[:,:10], labels[:,-1].reshape(-1,1)), axis=1)  # cut the 'null' class, which has second to last position
    if task['clip_eval_mode'] == 'sigmoid':
        if 'cholect45_triplet' in task['name']:
            threshold = f1max_thres(labels, preds)
            verbs_pred, targets_pred, instruments_pred = [], [], []
            verbs_gt, targets_gt, instruments_gt = [], [], []
            for pred in preds:
                verbs = np.zeros(len(dataset.verb_map))
                targets = np.zeros(len(dataset.target_map))
                instruments = np.zeros(len(dataset.instrument_map))
                # in pred vector get indices of 1s
                idx = np.where(pred > threshold)[0]  #get threshold from max f1 before splitting up predicitons into s,v,t
                for i in idx:
                    i_idx, v_idx, t_idx = dataset.maps[i]
                    verbs[v_idx] = 1
                    targets[t_idx] = 1
                    instruments[i_idx] = 1
                verbs_pred.append(verbs)
                targets_pred.append(targets)
                instruments_pred.append(instruments)
            for label in labels:
                verbs = np.zeros(len(dataset.verb_map))
                targets = np.zeros(len(dataset.target_map))
                instruments = np.zeros(len(dataset.instrument_map))
                idx = np.where(label == 1)[0]
                for i in idx:
                    i_idx, v_idx, t_idx = dataset.maps[i]
                    verbs[v_idx] = 1
                    targets[t_idx] = 1
                    instruments[i_idx] = 1
                verbs_gt.append(verbs)
                targets_gt.append(targets)
                instruments_gt.append(instruments)


            preds = [np.array(instruments_pred), np.array(verbs_pred), np.array(targets_pred)]
            labels = [np.array(instruments_gt), np.array(verbs_gt), np.array(targets_gt)]
            ### compute and display metrics
            print('Instruments:')
            map = map_for_classification(preds[0], labels[0])
            sliced_dict = dataset.instrument_map
            sliced_dict[6] = sliced_dict[-1]  # for some reason in the dataset the null class has key -1 for instruments but not verb and target
            del sliced_dict[-1]
            task['name'] = task['name'] + '_instrument'
            eval_metrics(labels[0], preds[0], sliced_dict, work_dir, task, model_name, name, len(labels[0]))
            print('Verb:')
            map = map_for_classification(preds[1], labels[1])
            task['name'] = task['name'].replace('_instrument', '_verb')
            eval_metrics(labels[1], preds[1], dataset.verb_map, work_dir, task, model_name, name, len(labels[1]))
            print('Target:')
            map = map_for_classification(preds[2], labels[2])
            task['name'] = task['name'].replace('_verb', '_target')
            eval_metrics(labels[2], preds[2], dataset.target_map, work_dir, task, model_name, name, len(labels[2]))
            return preds, labels
            ####################

        else:
            map = map_for_classification(labels, preds)
            threshold = f1max_thres(labels, preds)  # here it's one threshold overall, so resulting F1 is worse, but this seems more reasonable
            eval_metrics(labels, preds > threshold, label_map, work_dir, task, model_name, name, len(preds))
    elif task['clip_eval_mode'] == 'singlelabel' or task['clip_eval_mode'] == 'negative_examples':
        eval_metrics(labels, preds, label_map, work_dir, task, model_name, name, len(preds))


def eval_data(
    model, 
    work_dir, 
    name, 
    dataset, 
    task, 
    *args, 
    **kwargs
):
    # This function reads all files in a dataset and sets missing predictions to a dafault prediction.

    if isinstance(model.model, str):
        model_name = model.model
    elif hasattr(model, 'model_path'):
        model_name = model.model_path.split('/')[1]#VLM not API handled differently n VLMEvalKit
    else:
        model_name = model.name
    read_dir = osp.join(work_dir, task['name'], model_name, name)
    preds, labels = [], []
    successful_preds = 0
    evaluation_files = []
    all_files_labels = dict()
    all_files = []
    for frame, label in dataset.labels:
        out_file = osp.join(read_dir, f'{"-".join(frame.split("/")[-3:])}.json')
        all_files.append(out_file)
        all_files_labels[out_file] = label
        if osp.exists(out_file):
            evaluation_files.append(out_file)


    if 'dresden_anatomy' in task['name']:
        label_map = {idx: label for idx, label in enumerate(task['label_names'])}
        default = np.atleast_1d(np.zeros(len(label_map)-1))  # -1 for the 'null' class we ignore
        for file in tqdm(evaluation_files):
            with open(file, 'r') as f:
                pred = json.load(f)
            if "Blocked" in pred or "Exception" in pred:
                preds.append(default)
            elif len(pred) != len(default):
                preds.append(default)
            else:
                preds.append(np.array(list(pred.values())) * 1)
                successful_preds += 1

            label = all_files_labels[file]
            label_fixed = np.delete(label, -2)  # cut null class
            labels.append(label_fixed)

    elif 'error_recognition' in task['name']:
        label_map = {idx: label for idx, label in enumerate(task['label_names'])}
        labels_dict = {filename: label_array for filename, label_array in dataset.labels}
        default = 0
        for file in tqdm(evaluation_files):
            with open(file, 'r') as f:
                pred = json.load(f)
            if isinstance(pred, int):
                preds.append(pred)
            elif "Blocked" in pred or "Exception" in pred:
                preds.append(default)
            elif isinstance(pred, dict):
                preds.append(list(pred.values())[0])
                successful_preds += 1
            else:
                # get only the number using regex
                preds.append(int(re.search(r'\d+', pred).group()))
                successful_preds += 1

            label = task['label_names'].index(all_files_labels[file]) + 1
            labels.append(label)

    elif 'error_detection' in task['name']:
        extract_error_label = lambda x: (x[0], x[1])
        label_map = {}
        labels_dict = {path: (s, e) for path, (s, e, _) in dataset.labels}
        default = (0, 0)
        for file in tqdm(evaluation_files):
            with open(file, 'r') as f:
                pred = json.load(f)
            if "Blocked" in pred or "Exception" in pred:
                preds.append(default)
            elif isinstance(pred, dict):
                if 'gemini' in model_name:
                    start_time = pred['start_time']
                    end_time = pred['end_time']
                    start_time = (int(start_time.split(':')[0]) * 60 + int(start_time.split(':')[1])) * 10
                    end_time = (int(end_time.split(':')[0]) * 60 + int(end_time.split(':')[1])) * 10
                elif 'Qwen2-VL' in model_name:
                    start_time = pred['start']
                    end_time = pred['end']
                    if len(start_time.split(':')) == 2:
                        start_time = (int(start_time.split(':')[0]) * 60 + int(start_time.split(':')[1])) * 10
                    elif len(start_time.split(':')) == 3:
                        start_time = (int(start_time.split(':')[1]) * 60 + int(start_time.split(':')[2])) * 10
                    if len(end_time.split(':')) == 2:
                        end_time = (int(end_time.split(':')[0]) * 60 + int(end_time.split(':')[1])) * 10
                    elif len(end_time.split(':')) == 3:
                        end_time = (int(end_time.split(':')[1]) * 60 + int(end_time.split(':')[2])) * 10
                else:
                    if 'gpt-4o' in model_name:
                        nframes = 35
                    elif 'Phi-3.5-Vision' in model_name:
                        nframes = 35
                    elif 'Qwen2-VL' in model_name:
                        nframes = 35
                    elif 'InternVL2' in model_name:
                        nframes = 70
                    start_time = int(pred['start'] * (1800 / nframes))
                    end_time = int(pred['end'] * (1800 / nframes))
                preds.append((start_time, end_time))
                successful_preds += 1
            else:
                preds.append(default)
            
            label = extract_error_label(all_files_labels[file])
            labels.append(label)
        task['name'] = task['name'] + '_' + dataset.dataset_name

    elif 'multibypass140' in task['name']:
        label_map = {idx: label for idx, label in enumerate(task['label_names'])}
        default = np.atleast_1d(np.array(dataset.labels[0][1]) * 0)
        for file in tqdm(evaluation_files):
            with open(file, 'r') as f:
                pred = json.load(f)
            if "Blocked" in pred or "Exception" in pred:
                preds.append(default)
            else:
                preds.append(np.array(list(pred.values())) * 1)
                successful_preds += 1

            label = all_files_labels[file]
            labels.append(label)
        labels = np.array(labels).reshape(-1,1)

    elif 'cholec80' in task['name'] or 'cholect50' in task['name']:
        label_map = {idx: label for idx, label in enumerate(task['label_names'])}
        labels_dict = {filename: label_array for filename, label_array in dataset.labels}
        default = np.atleast_1d(np.array(dataset.labels[0][1]) * 0)
        for file in tqdm(evaluation_files):
            with open(file, 'r') as f:
                pred = json.load(f)
            if "Blocked" in pred or "Exception" in pred:
                preds.append(default)
            elif 'phase' in task['name']:
                preds.append(np.array(list(pred.values())) * 1)
                successful_preds += 1
            else:
                # Use key-based lookup so extra keys (e.g. SpecimenBag in CholecT50 prompts) are ignored
                pred_vec = np.array([pred.get(label, 0) for label in task['label_names']], dtype=np.float32)
                preds.append(pred_vec)
                successful_preds += 1
            label = all_files_labels[file]
            labels.append(label)
        if 'cholec80_phase_recognition' in task['name'] or 'cholect50_phase_recognition' in task['name']:
            labels = np.array(labels).reshape(-1,1)
    
    elif 'avos_action' in task['name']:
        label_map = {idx: label for idx, label in enumerate(task['label_names'])}
        label_map_inversed = {label: idx for idx, label in enumerate(task['label_names'])}
        default = np.atleast_1d(np.array(3))  # 3 is background
        for file in evaluation_files:
            with open(file, 'r') as f:
                pred = json.load(f)
            if not isinstance(pred, dict):
                preds.append(default)
            elif "Blocked" in pred or "Exception" in pred:
                preds.append(default)
            else:
                preds.append(np.array(list(pred.values())) * 1)
                successful_preds += 1
            folder = file.split('-')[2]
            frame = file.split('-')[3].strip('.json')
            label = label_map_inversed[all_files_labels[file]]
            labels.append(label)
        labels = np.array(labels).reshape(-1,1)
        print(' ')


    elif 'cholect45_triplet' in task['name']:
        label_map = {}
        offset = 0
        for d in [dataset.instrument_map, dataset.verb_map, dataset.target_map]:
            for key, value in d.items():
                if not value == 'null_instrument':
                    label_map[offset] = value
                    offset += 1
        verb_labels = get_triplet_component_labels(dataset.data_dir +'/verb')
        target_labels = get_triplet_component_labels(dataset.data_dir + '/target')
        instrument_labels = get_triplet_component_labels(dataset.data_dir + '/instrument')
        verbs, targets, instruments = [], [], []
        verbs_gt, targets_gt, instruments_gt = [], [], []
        default = {"instrument": ["null"], "verb": ["null"], "target": ["null"]}
        for file in tqdm(evaluation_files):  # only evaluate frames with predictions
            with open(file, 'r') as f:
                pred = json.load(f)
                if not isinstance(pred, dict) or "Blocked" in pred or "Exception" in pred:
                    pred = default
                elif 'instrument' not in pred or 'verb' not in pred or 'target' not in pred:
                    pred = default
                else:
                    successful_preds += 1
            verb_logits, target_logits, instrument_logits = pred_to_logits(pred, dataset)
            verbs.append(verb_logits)
            targets.append(target_logits)
            instruments.append(instrument_logits)
            fname = file.split('/')[-1]
            if fname[0] == 'r':
                folder = fname.split('-')[1]
                frame = str(int(fname.split('-')[2].split('.')[0]))
            else:
                folder = fname.split('-')[0]
                frame = str(int(fname.split('-')[1].split('.')[0]))
            verb_logits_gt, target_logits_gt, instrument_logits_gt = verb_labels[folder + '-' + frame], target_labels[folder + '-' + frame], instrument_labels[folder + '-' + frame]
            verbs_gt.append(verb_logits_gt)
            targets_gt.append(target_logits_gt)
            instruments_gt.append(instrument_logits_gt)

        preds = [np.array(instruments), np.array(verbs), np.array(targets)]
        labels = [np.array(instruments_gt), np.array(verbs_gt), np.array(targets_gt)]

    elif 'heichole' in task['name'] and 'skill_assessment' not in task['name']:
        label_map = {idx: label for idx, label in enumerate(task['label_names'])}
        default = np.atleast_1d(np.array(dataset.labels[0][1]) * 0)
        if 'heichole_tool_recognition' in task['name']:
                default = default[:7]
        for file in tqdm(evaluation_files):
            with open(file, 'r') as f:
                pred = json.load(f)
            if "Blocked" in pred or "Exception" in pred:
                preds.append(default)
            elif len(pred) != len(label_map) and 'phase' not in task['name']:
                preds.append(default)
            else:
                preds.append(np.array(list(pred.values())) * 1)
                successful_preds += 1
            label = all_files_labels[file]
            # HeiChole dataset states "Tools 7-20 Reserved for future additions" --> remove
            if 'heichole_tool_recognition' in task['name']:
                label = label[:7]
            labels.append(label)

    elif 'cvs' in task['name']:
        label_map = {idx: label for idx, label in enumerate(task['label_names'])}
        default = np.atleast_1d(np.array(dataset.labels[0][1]) * 0)

        for file in tqdm(evaluation_files):
            with open(file, 'r') as f:
                pred = json.load(f)
            if "Blocked" in pred or "Exception" in pred:
                preds.append(default)
            elif len(pred) != 3:
                preds.append(default)
            else:
                preds.append(np.array(list(pred.values())) * 1)
                successful_preds += 1
            label = all_files_labels[file]
            labels.append(label)
    
    elif 'endoscapes_object_detection' in task['name']: # TODO is this needed?
        label_map_inverted = dataset.category_ids_to_name
        label_map = {v: k for k, v in label_map_inverted.items()}
        labels_dict = {filename: label_array for filename, label_array in dataset.labels}
        cocoGt = COCO(osp.join(dataset.data_dir, dataset.split, 'annotation_coco.json'))
        image_ids_to_evaluate = []
        default = {}
        label_counts = {label: 0 for label in label_map.keys()}
        for file in evaluation_files:
            with open(file, 'r') as f:
                pred = json.load(f)
            if "Blocked" in pred or "Exception" in pred:
                pred = default
            else:
                successful_preds += 1
            folder = file.split('/')[-1].split('-')[2].split('_')[0]
            frame = file.split('/')[-1].split('_')[1].split('.')[0] + '.jpg'
            label = all_files_labels[file] # labels_dict[osp.join(dataset.data_dir, dataset.split, folder + '_' + frame)]
            image_id =  dataset.file_names_to_id[folder + '_' + frame]
            image_ids_to_evaluate.append(image_id)
            # add to label_counts
            for category_name in label.keys():
                if not category_name in label_map:
                    continue
                label_counts[category_name] += len(label[category_name])

            for category_name in pred.keys():
                if len(pred[category_name]) == 4:
                    y0, x0, y1, x1 = pred[category_name]
                else:
                    continue
                if 'gemini' in model_name:
                    x0 *= label['im_size_wh'][0] / 1000  #TODO this is for Gemini. Check how other models handle this
                    x1 *= label['im_size_wh'][0] / 1000
                    y0 *= label['im_size_wh'][1] / 1000
                    y1 *= label['im_size_wh'][1] / 1000
                elif 'paligemma' in model_name: # paligemma assumes coodinates for 1024 X 1024 images. endoscapes images are 854 x 480
                    x0 *= 854 / 1024
                    x1 *= 854 / 1024
                    y0 *= 480 / 1024
                    y1 *= 480 / 1024
                width = x1 - x0
                height = y1 - y0
                if not category_name.replace(' ','_') in label_map:
                    continue
                if not 'tool' in category_name:
                    category_id = label_map[category_name.replace('cyctic', 'cystic').replace(' ', '_')]
                    coco_format = {"image_id": image_id, "category_id": category_id, "bbox": [x0, y0, width, height], "score": 1.0}
                else:
                    coco_format = {"image_id": image_id, "category_id": label_map['tool'], "bbox": [x0, y0, width, height], "score": 1.0}
                preds.append(coco_format)
        dump(preds, osp.join(read_dir, 'results.json'))
        cocoDt = cocoGt.loadRes(osp.join(read_dir, 'results.json'))
        image_ids_to_evaluate = list(set(image_ids_to_evaluate))
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = image_ids_to_evaluate
        results = []
        if dataset.category == 'all':
            for category_id, category_name in dataset.category_ids_to_name.items():
                print('#####################################   Evaluating category:', category_name)
                cocoEval.params.catIds = [category_id]
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                # Extracting metrics
                AP_50_95_all = cocoEval.stats[0]  # AP@0.50:0.95 for area=all
                AP_50_all = cocoEval.stats[1]     # AP@0.50 for area=all
                AP_75_all = cocoEval.stats[2]     # AP@0.75 for area=all
                AR_1_all = cocoEval.stats[6]      # AR@1 for area=all
                AR_10_all = cocoEval.stats[7]     # AR@10 for area=all
                
                # Storing results for the category
                results.append({
                    'Class': category_name,
                    'AP@0.50:0.95': AP_50_95_all,
                    'AP@0.50': AP_50_all,
                    'AP@0.75': AP_75_all,
                    'AR@1': AR_1_all,
                    'AR@10': AR_10_all,
                })
        df = pd.DataFrame(results)

        # Separate "tool" and "anatomies"
        tool_df = df[df['Class'] == 'tool']
        anatomies_df = df[df['Class'] != 'tool']

        def weighted_average(df):
            metrics = df.select_dtypes(include=['float64']).columns
            total_weight = sum(label_counts.get(row['Class'], 0) for _, row in df.iterrows())
            weighted_metrics = {
                metric: sum(row[metric] * label_counts.get(row['Class'], 0) for _, row in df.iterrows()) / total_weight
                for metric in metrics
            }
            return weighted_metrics
        
        tool_weighted_avg = weighted_average(tool_df)
        tool_weighted_avg['Class'] = 'Weighted Average'

        anatomies_weighted_avg = weighted_average(anatomies_df)
        anatomies_weighted_avg['Class'] = 'Weighted Average'

        # Calculate average metrics
        tool_avg = tool_df.mean(numeric_only=True).to_dict()
        anatomies_avg = anatomies_df.mean(numeric_only=True).to_dict()

        # Add "Average" row
        tool_avg['Class'] = 'Average'
        tool_avg['Successful Preds'] = successful_preds  
        anatomies_avg['Class'] = 'Average'
        anatomies_avg['Successful Preds'] = successful_preds
        tool_df = pd.concat([tool_df, pd.DataFrame([tool_weighted_avg]), pd.DataFrame([tool_avg])], ignore_index=True).round(2)
        anatomies_df = pd.concat([anatomies_df, pd.DataFrame([anatomies_weighted_avg]), pd.DataFrame([anatomies_avg])], ignore_index=True).round(2)

        # Save to CSV
        if 'fewshot' in task['name']:
            tool_df.to_csv(work_dir + 'metrics_endoscapes_tool_detection_fewshot_' + model_name.replace('/','') + '_' + name + '.csv', index=False)
            anatomies_df.to_csv(work_dir + 'metrics_endoscapes_anatomy_detection_fewshot_' + model_name.replace('/','') + '_' + name + '.csv', index=False)
        else: 
            tool_df.to_csv(work_dir + 'metrics_endoscapes_tool_detection_' + model_name.replace('/','') + '_' + name + '.csv', index=False)
            anatomies_df.to_csv(work_dir + 'metrics_endoscapes_anatomy_detection_' + model_name.replace('/','') + '_' + name + '.csv', index=False)

        return preds, labels

    elif 'avos_object_detection' in task['name']:  # TODO is this needed?
        label_map_inverted = dataset.category_ids_to_name
        label_map = {v: k for k, v in label_map_inverted.items()}
        labels_dict = {filename: label_array for filename, label_array in dataset.labels}
        cocoGt = COCO('avos_coco_annotations.json')
        image_ids_to_evaluate = []
        default = {}
        label_counts = {label: 0 for label in label_map.keys()}
        frame_name_idx = 4
        if 'paligemma' in model_name:
            frame_name_idx = 5
        for file in tqdm(evaluation_files):
            with open(file, 'r') as f:
                pred = json.load(f)
            if "Blocked" in pred or "Exception" in pred:
                pred = default
            else:
                successful_preds += 1
            parts = file.split('-')
            frame = '-'.join(parts[frame_name_idx:]).split('.')[0] + '.jpg'
            label = all_files_labels[file] # labels_dict[osp.join(dataset.data_dir, dataset.split, folder + '_' + frame)]
            # add to label_counts
            for category_name in label.keys():
                if not category_name in label_map:
                    continue
                label_counts[category_name] += len(label[category_name])

            for category_name in pred.keys():
                if len(pred[category_name]) == 4:
                    y0, x0, y1, x1 = pred[category_name]
                    if isinstance(y0, dict):
                        continue
                    if isinstance(y0, str):
                        y0, x0, y1, x1 = int(y0), int(x0), int(y1), int(x1)
                else:
                    continue
                image_id =  dataset.file_names_to_id[frame]
                image_ids_to_evaluate.append(image_id)
                x0 *= label['im_size_wh'][0] / 1000  #TODO this is for Gemini. Check how other models handle this
                x1 *= label['im_size_wh'][0] / 1000
                y0 *= label['im_size_wh'][1] / 1000
                y1 *= label['im_size_wh'][1] / 1000
                width = x1 - x0
                height = y1 - y0
                if 'forceps' in category_name:
                    coco_format = {"image_id": image_id, "category_id": label_map['forceps'], "bbox": [x0, y0, width, height], "score": 1.0}
                elif 'bovie' in category_name:
                    coco_format = {"image_id": image_id, "category_id": label_map['bovie'], "bbox": [x0, y0, width, height], "score": 1.0}
                elif 'needledriver' in category_name:
                    coco_format = {"image_id": image_id, "category_id": label_map['needledriver'], "bbox": [x0, y0, width, height], "score": 1.0}
                elif 'hand' in category_name:
                    coco_format = {"image_id": image_id, "category_id": label_map['hand'], "bbox": [x0, y0, width, height], "score": 1.0}
                preds.append(coco_format)
        dump(preds, osp.join(read_dir, 'results.json'))
        cocoDt = cocoGt.loadRes(osp.join(read_dir, 'results.json'))
        image_ids_to_evaluate = list(set(image_ids_to_evaluate))
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = image_ids_to_evaluate
        results = []
        if dataset.category == 'all':
            for category_id, category_name in dataset.category_ids_to_name.items():
                print('#####################################   Evaluating category:', category_name)
                cocoEval.params.catIds = [category_id]
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                # Extracting metrics
                AP_50_95_all = cocoEval.stats[0]  # AP@0.50:0.95 for area=all
                AP_50_all = cocoEval.stats[1]     # AP@0.50 for area=all
                AP_75_all = cocoEval.stats[2]     # AP@0.75 for area=all
                AR_1_all = cocoEval.stats[6]      # AR@1 for area=all
                AR_10_all = cocoEval.stats[7]     # AR@10 for area=all
                
                # Storing results for the category
                results.append({
                    'Class': category_name,
                    'AP@0.50:0.95': AP_50_95_all,
                    'AP@0.50': AP_50_all,
                    'AP@0.75': AP_75_all,
                    'AR@1': AR_1_all,
                    'AR@10': AR_10_all,
                })
        df = pd.DataFrame(results)

        # Separate "tool" and "anatomies"
        hand_df = df[df['Class'] == 'hand']
        tools_df = df[df['Class'] != 'hand']

        def weighted_average(df):
            metrics = df.select_dtypes(include=['float64']).columns
            total_weight = sum(label_counts.get(row['Class'], 0) for _, row in df.iterrows())
            weighted_metrics = {
                metric: sum(row[metric] * label_counts.get(row['Class'], 0) for _, row in df.iterrows()) / total_weight
                for metric in metrics
            }
            return weighted_metrics
        
        hand_weighted_avg = weighted_average(hand_df)
        hand_weighted_avg['Class'] = 'Weighted Average'

        tools_weighted_avg = weighted_average(tools_df)
        tools_weighted_avg['Class'] = 'Weighted Average'

        # Calculate average metrics
        hand_avg = hand_df.mean(numeric_only=True).to_dict()
        tools_avg = tools_df.mean(numeric_only=True).to_dict()

        # Add "Average" row
        hand_avg['Class'] = 'Average'
        hand_avg['Successful Preds'] = successful_preds  
        tools_avg['Class'] = 'Average'
        tools_avg['Successful Preds'] = successful_preds
        hand_df = pd.concat([hand_df, pd.DataFrame([hand_weighted_avg]), pd.DataFrame([hand_avg])], ignore_index=True).round(2)
        tools_df = pd.concat([tools_df, pd.DataFrame([tools_weighted_avg]), pd.DataFrame([tools_avg])], ignore_index=True).round(2)

        # Save to CSV
        hand_df.to_csv(work_dir + 'metrics_avos_hand_detection_' + model_name.replace('/','') + '_' + name + '.csv', index=False)
        tools_df.to_csv(work_dir + 'metrics_avos_tools_detection_' + model_name.replace('/','') + '_' + name + '.csv', index=False)

        return preds, labels
    
    elif 'jigsaws' in task['name'] or 'autolaparo' in task['name'] or 'heichole_skill_assessment' in task['name']:
        label_map = {idx: label for idx, label in enumerate(task['label_names'])}
        successful_preds = 0
        for i, pred in enumerate(preds):
            pattern = '|'.join([re.escape(label) for label in task.label_names])
            matches = re.findall(pattern, pred)
            if len(matches) == 1:
                preds[i] = matches[0]
                successful_preds += 1
            elif len(matches) == 0:
                preds[i] = task.label_names[0]
            else:
                preds[i] = matches[-1]
                successful_preds += 1
        preds = np.array(preds)
        labels = np.array(labels)


    ### compute and display metrics
    if 'triplet' in task['name']:
        print('Instruments:')
        map = map_for_classification(preds[0], labels[0])
        sliced_dict = {k: label_map[k] for k in range(preds[0].shape[1])}
        task['name'] = task['name'] + '_instrument'
        eval_metrics(labels[0], preds[0], sliced_dict, work_dir, task, model_name, name, successful_preds)
        print('Verb:')
        map = map_for_classification(preds[1], labels[1])
        sliced_dict = {k-preds[0].shape[1]: label_map[k] for k in range(preds[0].shape[1], preds[1].shape[1]+ preds[0].shape[1])}
        task['name'] = task['name'].replace('_instrument', '_verb')
        eval_metrics(labels[1], preds[1], sliced_dict, work_dir, task, model_name, name, successful_preds)
        print('Target:')
        map = map_for_classification(preds[2], labels[2])
        sliced_dict = {k-preds[0].shape[1]-preds[1].shape[1]: label_map[k] for k in range(preds[0].shape[1] + preds[1].shape[1], preds[2].shape[1] + preds[0].shape[1] + preds[1].shape[1])}
        task['name'] = task['name'].replace('_verb', '_target')
        eval_metrics(labels[2], preds[2], sliced_dict, work_dir, task, model_name, name, successful_preds)
        return preds, labels

    if not 'error_detection' in task['name']:
        preds = np.array(preds, dtype=np.uint8)
        labels = np.array(labels, dtype=np.uint8)
    eval_metrics(labels, preds, label_map, work_dir, task, model_name, name, successful_preds, len_test_files=len(all_files))

    return preds, labels


def eval_metrics(
    labels, 
    preds, 
    label_map, 
    work_dir, 
    task, 
    model_name, 
    name,
    successful_preds,
    len_test_files=None
):
    print(f'Finished evaluating {model_name}')
    if not 'phase' in task['name'] \
        and not 'avos' in task['name'] \
        and not 'error_detection' in task['name'] \
        and not 'error_recognition' in task['name'] \
        and not 'disease_severity' in task['name'] \
        and not 'intermountain_skill_assessment' in task['name']:
        for i in range(labels.shape[1]):
            print(f'Instances of {label_map[i]}: {np.sum(labels[:, i])}')
        print('Total instances:', labels.shape[0])
        for i in range(preds.shape[1]):
            print(f'Predicted instances of {label_map[i]}: {np.sum(preds[:, i])}')

    if 'phase_recognition' in task['name'] or 'avos' in task['name'] or 'gesture_classification' in task['name']:
        out  = np.unique(labels, axis=0, return_counts=True)
        total_numbers_per_class = out[1]
        print('total number of labels per class: ', total_numbers_per_class)
    elif 'disease_severity' in task['name'] or 'intermountain_skill_assessment' in task['name']: # TODO is this needed?
        total_numbers_per_class = np.unique(labels, return_counts=True)[1]
        print(total_numbers_per_class)
    else:
        total_numbers_per_class = np.sum(labels, axis=0)

    if 'intermountain_skill_assessment' in task['name']: # TODO is this needed?
        f1_values = f1(labels, preds, average=None)
        print('f1_values: ', f1_values)
        accuracy_macro = accuracy(labels, preds)
        metrics_df = pd.DataFrame({
            'Class': ['Average'],
            'F1 Score': [np.mean(f1_values)],
            'Accuracy': [accuracy_macro],
            'Successful Preds': [successful_preds]
        })
        print(metrics_df.to_string(index=False))
        print('-'*60)
        metrics_df.to_csv(work_dir + 'metrics_' + task['name'] + '_' + model_name.replace('/','') + '_' + name + '.csv', index=False)
        return preds, labels

    if 'error_detection' in task['name']:
        metrics_df = pd.DataFrame({
            'Class': ['Average'],
            'mIoU': [mloc_iou(labels, preds)],
            'Successful Preds': [successful_preds]
        })
        print(metrics_df.to_string(index=False))
        print('-'*60)
        metrics_df.to_csv(work_dir + 'metrics_' + task['name'] + '_' + model_name.replace('/','') + '_' + name + '.csv', index=False)
        return preds, labels

    ## recall, precision, f1, jaccard
    recall_values = recall(labels, preds, average=None)
    precision_values = precision(labels, preds,average=None)
    jaccard_values = jaccard(labels, preds, average=None)
    f1_values = f1(labels, preds, average=None)
    recall_micro = recall(labels, preds, average='micro')
    precision_micro = precision(labels, preds, average='micro')
    jaccard_micro = jaccard(labels, preds, average='micro')
    f1_micro = f1(labels, preds, average='micro')
    accuracy_macro = accuracy(labels.flatten(), preds.flatten())

    # weighted average
    recall_avg = recall(labels, preds, average='weighted')
    precision_avg = precision(labels, preds, average='weighted')
    jaccard_avg = jaccard(labels, preds, average='weighted')
    f1_avg = f1(labels, preds, average='weighted')

    if 'dresden_anatomy' in task['name']: # TODO is this needed?
        ### remove null class
        null_index = next(key for key, value in label_map.items() if value == 'null')
        map_ids = [i for i in range(len(label_map)) if i != null_index]
    elif 'phase_recognition' in task['name'] or 'avos' in task['name']:
        if np.ndim(labels) != np.ndim(preds):
            if np.ndim(labels) == 1:
                labels = labels.reshape(-1,1)
            elif np.ndim(preds) == 1:
                preds = preds.reshape(-1,1)
        map_ids = np.unique([labels, preds])
    elif 'error_recognition' in task['name']:
        map_ids = np.unique([labels, preds]) - 1
    else:
        map_ids = range(len(label_map))

    # printing/saving
    print('Metrics:')
    metrics_df = pd.DataFrame({
        'Class': [f'{label_map[i]}' for i in map_ids],
        'Recall': recall_values,
        'Precision': precision_values,
        'Jaccard': jaccard_values,
        'F1 Score': f1_values
    })
    weighted_avg_df = pd.DataFrame({
        'Class': ['Weighted Average'],
        'Recall': [recall_avg],
        'Precision': [precision_avg],
        'Jaccard': [jaccard_avg],
        'F1 Score': [f1_avg]
    })
    avg_df = pd.DataFrame({
        'Class': ['Average'],
        'Recall': [np.mean(recall_values)],
        'Precision': [np.mean(precision_values)],
        'Jaccard': [np.mean(jaccard_values)],
        'F1 Score': [np.mean(f1_values)],
        'Accuracy': [accuracy_macro],
        'Successful Preds': [successful_preds]
    })
    metrics_df = pd.concat([metrics_df, weighted_avg_df, avg_df], ignore_index=True)
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(work_dir + 'metrics_' + task['name'] + '_' + model_name.replace('/','') + '_' + name + '.csv', index=False)
    return preds, labels


# ── CholecT50 Status Reasoning ──────────────────────────────────────────────

PHASE_COT = {
    "Preparation": {
        "early": "Trocar insertion, initial camera positioning, initial exposure of the surgical field.",
        "middle": "Grasper positioning on gallbladder fundus, initial retraction to expose Calot's triangle area.",
        "late": "Good retraction established, gallbladder lifted, surgical field fully exposed and ready for dissection.",
    },
    "Calot Triangle Dissection": {
        "early": "Initial dissection near the gallbladder neck, hook or dissector approaching Calot's triangle.",
        "middle": "Progressive dissection clearing fat and tissue from Calot's triangle, cystic duct partially exposed.",
        "late": "Cystic duct and artery fully exposed and isolated, Critical View of Safety nearly or fully achieved.",
    },
    "Clipping & Cutting": {
        "early": "Clipper introduced, approaching cystic duct for first clip placement.",
        "middle": "Cystic duct clipped, moving to clip the cystic artery.",
        "late": "Both structures clipped and divided, scissors completing final cuts.",
    },
    "Gallbladder Dissection": {
        "early": "Hook beginning to dissect gallbladder from the liver bed, starting at the neck.",
        "middle": "Gallbladder partially detached, dissection progressing along the liver bed with coagulation as needed.",
        "late": "Gallbladder nearly fully detached, only thin attachments remaining at the fundus.",
    },
    "Gallbladder Packaging": {
        "early": "Specimen bag being introduced into the abdomen.",
        "middle": "Gallbladder being maneuvered toward the specimen bag opening.",
        "late": "Gallbladder placed inside the bag, bag being closed or retrieved.",
    },
    "Cleaning & Coagulation": {
        "early": "Irrigator washing the liver bed, initial inspection for bleeding points.",
        "middle": "Active coagulation of bleeding spots with bipolar, irrigation and aspiration of fluid.",
        "late": "Liver bed appears dry, final inspection with minimal active bleeding.",
    },
    "Gallbladder Retraction": {
        "early": "Specimen bag with gallbladder being pulled toward the trocar site.",
        "middle": "Extraction through trocar port, potentially enlarging incision.",
        "late": "Gallbladder fully extracted, instruments being removed.",
    },
}


def _fill_status_prompt(template, phase, few_shot_examples, task_name):
    """Fill prompt template with phase-specific CoT and few-shot examples."""
    cot = PHASE_COT.get(phase, {"early": "...", "middle": "...", "late": "..."})

    examples_text = ""
    for i, (_, _, ex_meta) in enumerate(few_shot_examples):
        if 'progress' in task_name:
            examples_text += f"\n  Example {i+1}: progress={ex_meta['progress']}% -> {{\"progress\": {ex_meta['progress']}}}"
        elif 'tick' in task_name:
            ex_next = ex_meta.get('next_phase', ex_meta['phase'])
            same_or_diff = "(phase ongoing)" if ex_next == ex_meta['phase'] else "(phase transitioning)"
            examples_text += (
                f"\n  Example {i+1} {same_or_diff}: current_phase='{ex_meta['phase']}', "
                f"next_phase='{ex_next}'"
            )

    return template.format(
        phase=phase,
        cot_early=cot['early'],
        cot_middle=cot['middle'],
        cot_late=cot['late'],
        few_shot_examples=examples_text,
    )


def infer_data_status(
    model,
    work_dir,
    name,
    dataset,
    task,
    prompt_template,
    **kwargs
):
    """Inference for cholect50_phase_planning tasks (multi-frame)."""
    if isinstance(model.model, str):
        model_name = model.model
    elif hasattr(model, 'model_path'):
        model_name = model.model_path.split('/')[1]
    else:
        model_name = model.name

    write_dir = osp.join(work_dir, task['name'], model_name, name)
    if not osp.exists(write_dir):
        os.makedirs(write_dir)

    task_name = task['name']

    # Prompt is now static (no per-sample placeholders) — save it once
    with open(osp.join(write_dir, 'prompt.txt'), 'w') as f:
        f.write(prompt_template)

    for frame, label in tqdm(dataset):
        meta = frame['meta']
        out_file = osp.join(write_dir, f'{meta["id"]}.json')
        if osp.exists(out_file) and not kwargs.get('override_outputs', False):
            continue

        if osp.exists(out_file):
            os.remove(out_file)

        # Build message: multiple frames + prompt (same prompt for all samples)
        if 'paths' in frame:
            message = frame['paths'] + [prompt_template]
        else:
            message = [frame['path'], prompt_template]

        try:
            ret = model.generate(message)
            if isinstance(ret, int):
                dump({'answer': None, 'raw_response': f'Blocked: {ret}'}, out_file)
                continue

            # Extract JSON from response
            start_index = ret.find('{')
            end_index = ret.rfind('}')
            if start_index != -1 and end_index != -1:
                result = ret[start_index:end_index + 1]
            else:
                result = '{}'
            result = result.strip('`').strip('json').replace('\n', '')
            result = result.replace('False', '0').replace('True', '1').replace('false', '0').replace('true', '1')
            parsed = json.loads(result)

            # Save concise output: {"answer": ..., "raw_response": ...}
            if 'progress_prediction' in task_name:
                answer = {
                    'current_phase': parsed.get('current_phase', ''),
                    'progress': parsed.get('progress', ''),
                }
            else:
                answer = {
                    'current_phase': parsed.get('current_phase', ''),
                    'next_phase': parsed.get('next_phase', ''),
                }

            dump({'answer': answer, 'raw_response': ret}, out_file)

        except Exception as e:
            print(f'Exception: {e}')
            dump({'answer': None, 'raw_response': f'Exception: {e}'}, out_file)
            continue

    # Evaluate
    preds, labels = eval_data_status(model, work_dir, name, dataset, task)
    return preds, labels


def eval_data_status(
    model,
    work_dir,
    name,
    dataset,
    task,
    *args,
    **kwargs
):
    """Evaluate cholect50_phase_planning predictions."""
    if isinstance(model.model, str):
        model_name = model.model
    elif hasattr(model, 'model_path'):
        model_name = model.model_path.split('/')[1]
    else:
        model_name = model.name

    read_dir = osp.join(work_dir, task['name'], model_name, name)
    task_name = task['name']

    successful = 0
    all_preds = []
    all_labels = []
    all_metas = []

    for idx in range(len(dataset)):
        frame_paths, label, meta = dataset.labels[idx]
        out_file = osp.join(read_dir, f'{meta["id"]}.json')

        if not osp.exists(out_file):
            all_preds.append(None)
            all_labels.append(label)
            all_metas.append(meta)
            continue

        with open(out_file, 'r') as f:
            result = json.load(f)

        answer = result.get('answer', None)
        all_preds.append(answer)
        all_labels.append(label)
        all_metas.append(meta)
        if answer is not None:
            successful += 1

    # Compute metrics
    valid_cur = [(p, m) for p, m in zip(all_preds, all_metas) if p is not None and isinstance(p, dict)]

    if not valid_cur:
        metrics_df = pd.DataFrame({'Metric': ['Error'], 'Value': ['No valid predictions']})
    elif 'progress_prediction' in task_name:
        cur_phase_correct = sum(1 for p, m in valid_cur if p.get('current_phase', '').lower() == m['phase'].lower())
        progress_correct = sum(1 for p, m in valid_cur if p.get('progress', '').lower() == m.get('progress_class', '').lower())
        # Per-class accuracy
        from collections import Counter
        class_correct = Counter()
        class_total = Counter()
        for p, m in valid_cur:
            gt_class = m.get('progress_class', '')
            class_total[gt_class] += 1
            if p.get('progress', '').lower() == gt_class.lower():
                class_correct[gt_class] += 1
        metrics_df = pd.DataFrame({
            'Metric': ['Current Phase Acc', 'Progress Acc',
                       'Early Acc', 'Middle Acc', 'Late Acc', 'Valid', 'Total'],
            'Value': [
                round(cur_phase_correct / len(valid_cur), 3),
                round(progress_correct / len(valid_cur), 3),
                round(class_correct['early'] / max(class_total['early'], 1), 3),
                round(class_correct['middle'] / max(class_total['middle'], 1), 3),
                round(class_correct['late'] / max(class_total['late'], 1), 3),
                len(valid_cur),
                len(all_preds),
            ]
        })
    else:
        cur_correct = sum(1 for p, m in valid_cur if p.get('current_phase', '').lower() == m['phase'].lower())
        next_correct = sum(1 for p, m in valid_cur if p.get('next_phase', '').lower() == m.get('next_phase', m['phase']).lower())
        metrics_df = pd.DataFrame({
            'Metric': ['Current Phase Acc', 'Next Phase Acc', 'Valid', 'Total'],
            'Value': [
                round(cur_correct / len(valid_cur), 3),
                round(next_correct / len(valid_cur), 3),
                len(valid_cur),
                len(all_preds),
            ]
        })

    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(osp.join(work_dir, f'metrics_{task_name}_{model_name.replace("/","")}_{name}.csv'), index=False)
    return all_preds, all_labels


def _triplet_set(tlist):
    """Convert list of triplet dicts to a set of (instrument, verb, target) tuples."""
    if not tlist:
        return set()
    s = set()
    for t in tlist:
        if isinstance(t, dict):
            s.add((t.get('instrument', ''), t.get('verb', ''), t.get('target', '')))
    return s


def _set_f1(gt_set, pred_set):
    if not gt_set and not pred_set:
        return 1.0
    if not gt_set or not pred_set:
        return 0.0
    tp = len(gt_set & pred_set)
    p = tp / len(pred_set)
    r = tp / len(gt_set)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _component_f1(gt_set, pred_set, idx):
    a = set(t[idx] for t in gt_set) if gt_set else set()
    b = set(t[idx] for t in pred_set) if pred_set else set()
    return _set_f1(a, b)


def infer_data_triplet_planning(
    model,
    work_dir,
    name,
    dataset,
    task,
    prompt_template,
    **kwargs
):
    """Inference for cholect50_phase_triplet_planning (multi-frame, JSON output)."""
    if isinstance(model.model, str):
        model_name = model.model
    elif hasattr(model, 'model_path'):
        model_name = model.model_path.split('/')[1]
    else:
        model_name = model.name

    write_dir = osp.join(work_dir, task['name'], model_name, name)
    if not osp.exists(write_dir):
        os.makedirs(write_dir)

    with open(osp.join(write_dir, 'prompt.txt'), 'w') as f:
        f.write(prompt_template)

    for frame, label in tqdm(dataset):
        meta = frame['meta']
        out_file = osp.join(write_dir, f'{meta["id"]}.json')
        if osp.exists(out_file) and not kwargs.get('override_outputs', False):
            continue

        if osp.exists(out_file):
            os.remove(out_file)

        if 'paths' in frame:
            message = frame['paths'] + [prompt_template]
        else:
            message = [frame['path'], prompt_template]

        try:
            ret = model.generate(message)
            if isinstance(ret, int):
                dump({'answer': None, 'raw_response': f'Blocked: {ret}'}, out_file)
                continue

            # Extract JSON
            start_index = ret.find('{')
            end_index = ret.rfind('}')
            if start_index != -1 and end_index != -1:
                result = ret[start_index:end_index + 1]
            else:
                result = '{}'
            result = result.strip('`').strip('json').replace('\n', '')
            result = result.replace('False', '0').replace('True', '1').replace('false', '0').replace('true', '1')
            parsed = json.loads(result)

            answer = {
                'phase': parsed.get('phase', ''),
                'current_triplet': parsed.get('current_triplet', []),
                'next_phase': parsed.get('next_phase', ''),
                'next_triplet': parsed.get('next_triplet', []),
            }

            dump({'answer': answer, 'raw_response': ret}, out_file)

        except Exception as e:
            print(f'Exception: {e}')
            dump({'answer': None, 'raw_response': f'Exception: {e}'}, out_file)
            continue

    preds, labels = eval_data_triplet_planning(model, work_dir, name, dataset, task)
    return preds, labels


def eval_data_triplet_planning(
    model,
    work_dir,
    name,
    dataset,
    task,
    *args,
    **kwargs
):
    """Evaluate cholect50_phase_triplet_planning predictions."""
    if isinstance(model.model, str):
        model_name = model.model
    elif hasattr(model, 'model_path'):
        model_name = model.model_path.split('/')[1]
    else:
        model_name = model.name

    read_dir = osp.join(work_dir, task['name'], model_name, name)
    task_name = task['name']

    all_preds = []
    all_labels = []
    all_metas = []

    for idx in range(len(dataset)):
        frame_paths, label, meta = dataset.labels[idx]
        out_file = osp.join(read_dir, f'{meta["id"]}.json')

        if not osp.exists(out_file):
            all_preds.append(None)
            all_labels.append(label)
            all_metas.append(meta)
            continue

        with open(out_file, 'r') as f:
            result = json.load(f)

        all_preds.append(result.get('answer', None))
        all_labels.append(label)
        all_metas.append(meta)

    # Compute metrics
    valid = [(p, m) for p, m in zip(all_preds, all_metas) if p is not None and isinstance(p, dict)]
    if not valid:
        metrics_df = pd.DataFrame({'Metric': ['Error'], 'Value': ['No valid predictions']})
        print(metrics_df.to_string(index=False))
        return all_preds, all_labels

    n = len(valid)

    # Phase accuracy
    phase_correct = sum(1 for p, m in valid if p.get('phase', '').lower() == m['phase'].lower())

    # Next phase accuracy
    next_phase_correct = sum(1 for p, m in valid
                             if p.get('next_phase', '').lower() == m['next_phase'].lower())

    # Current triplet metrics
    ct_inst_f1 = sum(_component_f1(_triplet_set(m['current_triplet']),
                                    _triplet_set(p.get('current_triplet', [])), 0)
                     for p, m in valid) / n
    ct_verb_f1 = sum(_component_f1(_triplet_set(m['current_triplet']),
                                    _triplet_set(p.get('current_triplet', [])), 1)
                     for p, m in valid) / n
    ct_targ_f1 = sum(_component_f1(_triplet_set(m['current_triplet']),
                                    _triplet_set(p.get('current_triplet', [])), 2)
                     for p, m in valid) / n

    # Next triplet metrics
    nt_inst_f1 = sum(_component_f1(_triplet_set(m['next_triplet']),
                                    _triplet_set(p.get('next_triplet', [])), 0)
                     for p, m in valid) / n
    nt_verb_f1 = sum(_component_f1(_triplet_set(m['next_triplet']),
                                    _triplet_set(p.get('next_triplet', [])), 1)
                     for p, m in valid) / n
    nt_targ_f1 = sum(_component_f1(_triplet_set(m['next_triplet']),
                                    _triplet_set(p.get('next_triplet', [])), 2)
                     for p, m in valid) / n
    nt_exact = sum(1 for p, m in valid
                   if _triplet_set(m['next_triplet']) == _triplet_set(p.get('next_triplet', [])))

    metrics_df = pd.DataFrame({
        'Metric': [
            'Phase Acc', 'Next Phase Acc',
            'Cur Triplet Inst F1', 'Cur Triplet Verb F1', 'Cur Triplet Targ F1',
            'Next Triplet Inst F1', 'Next Triplet Verb F1', 'Next Triplet Targ F1',
            'Next Triplet Exact', 'Valid', 'Total',
        ],
        'Value': [
            round(phase_correct / n, 3),
            round(next_phase_correct / n, 3),
            round(ct_inst_f1, 3), round(ct_verb_f1, 3), round(ct_targ_f1, 3),
            round(nt_inst_f1, 3), round(nt_verb_f1, 3), round(nt_targ_f1, 3),
            round(nt_exact / n, 3),
            n, len(all_preds),
        ]
    })

    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(osp.join(work_dir, f'metrics_{task_name}_{model_name.replace("/","")}_{name}.csv'), index=False)
    return all_preds, all_labels


def infer_data_vtrb_recognition(
    model,
    work_dir,
    name,
    dataset,
    task,
    prompt_template,
    **kwargs
):
    """Inference for VTRB-Suturing few-shot bbox recognition.

    Each sample sends: few-shot images with annotation text + test image + query.
    """
    if isinstance(model.model, str):
        model_name = model.model
    elif hasattr(model, 'model_path'):
        model_name = model.model_path.split('/')[1]
    else:
        model_name = model.name

    write_dir = osp.join(work_dir, task['name'], model_name, name)
    if not osp.exists(write_dir):
        os.makedirs(write_dir)

    with open(osp.join(write_dir, 'prompt.txt'), 'w') as f:
        f.write(prompt_template)

    for frame, label in tqdm(dataset):
        meta = frame['meta']
        out_file = osp.join(write_dir, f'{meta["id"]}.json')
        if osp.exists(out_file) and not kwargs.get('override_outputs', False):
            continue

        if osp.exists(out_file):
            os.remove(out_file)

        # Build few-shot message: image + annotation text for each example
        few_shot = meta.get('few_shot_examples', [])

        # Check if model supports interleaved multi-image
        is_interleave = getattr(model, 'INTERLEAVE', True)
        if hasattr(model, 'model') and hasattr(model.model, 'INTERLEAVE'):
            is_interleave = model.model.INTERLEAVE
        # PaliGemma and similar single-image models
        model_name_lower = model_name.lower()
        if 'paligemma' in model_name_lower or 'pali' in model_name_lower:
            is_interleave = False

        if is_interleave:
            message = []
            message.append(
                'I will show you example images with bounding box annotations for a robotic suturing scene.\n'
                'Classes: grippers, target_tissue, wound_gap, bites, needle.\n'
                'Bounding boxes are [x1%, y1%, x2%, y2%] as percentage of image (0-100).\n\n'
            )
            for i, ex in enumerate(few_shot):
                message.append(ex['image_path'])
                objects_list = []
                for cls, bboxes in ex['objects'].items():
                    for b in bboxes:
                        objects_list.append({'class': cls, 'bbox': b['bbox']})
                message.append(
                    f'Example {i+1} annotations:\n'
                    f'{json.dumps(objects_list, indent=2)}\n\n'
                )
            message.append(prompt_template)
            message.append(meta['test_image_path'])
        else:
            # Single-image model: send text-only few-shot + single test image
            fs_text = (
                'Detect objects in this robotic suturing image.\n'
                'Classes: grippers, target_tissue, wound_gap, bites, needle.\n'
                'Output bounding boxes as [x1%, y1%, x2%, y2%] percentage (0-100).\n\n'
                'Reference annotations from similar images:\n'
            )
            for i, ex in enumerate(few_shot):
                objects_list = []
                for cls, bboxes in ex['objects'].items():
                    for b in bboxes:
                        objects_list.append({'class': cls, 'bbox': b['bbox']})
                fs_text += f'Example {i+1}: {json.dumps(objects_list)}\n'
            fs_text += '\nDetect all objects in this image. '
            fs_text += 'Respond in JSON: {"objects": [{"class": "...", "bbox": [x1, y1, x2, y2]}]}'
            message = [meta['test_image_path'], fs_text]

        try:
            ret = model.generate(message)
            if isinstance(ret, int):
                dump({'answer': None, 'raw_response': f'Blocked: {ret}'}, out_file)
                continue

            # Extract JSON or parse <loc> tokens
            parsed = None

            # Try PaliGemma <loc> format first: <locXXXX><locYYYY>... class_name
            import re
            loc_pattern = re.compile(r'<loc(\d{4})>')
            loc_matches = loc_pattern.findall(ret)
            if loc_matches and len(loc_matches) >= 4:
                # Parse PaliGemma loc tokens (0-1023 scale) → percentage (0-100)
                objects = []
                # Split by class labels — format: <loc><loc><loc><loc> class ; ...
                segments = re.split(r'\s*;\s*', ret.strip())
                for seg in segments:
                    locs = [int(x) for x in loc_pattern.findall(seg)]
                    # Remove loc tokens to get class name
                    cls_name = loc_pattern.sub('', seg).strip().lower()
                    cls_name = cls_name.strip('. ,;')
                    if len(locs) >= 4 and cls_name:
                        y1_pct = round(locs[0] / 1024 * 100, 1)
                        x1_pct = round(locs[1] / 1024 * 100, 1)
                        y2_pct = round(locs[2] / 1024 * 100, 1)
                        x2_pct = round(locs[3] / 1024 * 100, 1)
                        objects.append({
                            'class': cls_name,
                            'bbox': [x1_pct, y1_pct, x2_pct, y2_pct],
                        })
                if objects:
                    parsed = {'objects': objects}

            # Fall back to JSON extraction
            if parsed is None:
                start_index = ret.find('{')
                end_index = ret.rfind('}')
                if start_index != -1 and end_index != -1:
                    try:
                        json_str = ret[start_index:end_index + 1]
                        json_str = json_str.replace('\n', ' ')
                        json_str = json_str.replace('True', 'true').replace('False', 'false')
                        parsed = json.loads(json_str)
                    except json.JSONDecodeError:
                        # Fix common malformed JSON: incomplete bbox arrays, trailing commas
                        import re
                        fixed = json_str
                        # Remove trailing commas before ] or }
                        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
                        # Fix bbox arrays with < 4 values: pad with 0
                        def fix_bbox(m):
                            nums = [x.strip() for x in m.group(1).split(',') if x.strip()]
                            while len(nums) < 4:
                                nums.append('0')
                            return '"bbox": [' + ', '.join(nums[:4]) + ']'
                        fixed = re.sub(r'"bbox":\s*\[([\d., ]*)\]', fix_bbox, fixed)
                        try:
                            parsed = json.loads(fixed)
                        except json.JSONDecodeError:
                            pass

            dump({'answer': parsed, 'raw_response': ret}, out_file)

        except Exception as e:
            print(f'Exception: {e}')
            dump({'answer': None, 'raw_response': f'Exception: {e}'}, out_file)
            continue

    # Eval — just count valid responses (no GT)
    eval_data_vtrb_recognition(model, work_dir, name, dataset, task)


def eval_data_vtrb_recognition(
    model,
    work_dir,
    name,
    dataset,
    task,
    *args,
    **kwargs
):
    """Evaluate VTRB recognition — count valid responses and detection stats."""
    if isinstance(model.model, str):
        model_name = model.model
    elif hasattr(model, 'model_path'):
        model_name = model.model_path.split('/')[1]
    else:
        model_name = model.name

    read_dir = osp.join(work_dir, task['name'], model_name, name)
    task_name = task['name']

    total = 0
    valid = 0
    total_objects = 0
    class_counts = {}

    for idx in range(len(dataset)):
        _, _, meta = dataset.labels[idx]
        out_file = osp.join(read_dir, f'{meta["id"]}.json')
        total += 1

        if not osp.exists(out_file):
            continue

        with open(out_file) as f:
            result = json.load(f)

        answer = result.get('answer')
        if answer is None:
            continue

        objects = answer.get('objects', [])
        if objects:
            valid += 1
            total_objects += len(objects)
            for obj in objects:
                cls = obj.get('class', 'unknown')
                class_counts[cls] = class_counts.get(cls, 0) + 1

    avg_objects = round(total_objects / valid, 1) if valid > 0 else 0
    metrics_df = pd.DataFrame({
        'Metric': ['Valid Responses', 'Total', 'Avg Objects/Frame'] +
                  [f'Count: {cls}' for cls in sorted(class_counts.keys())],
        'Value': [valid, total, avg_objects] +
                 [class_counts[cls] for cls in sorted(class_counts.keys())],
    })

    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(osp.join(work_dir, f'metrics_{task_name}_{model_name.replace("/","")}_{name}.csv'), index=False)


# ── MCQ Tasks (per-sample choices) ──────────────────────────────────────────

def infer_data_mcq(
    model,
    work_dir,
    name,
    dataset,
    task,
    prompt_template,
    **kwargs
):
    """Inference for MCQ tasks with per-sample choices (e.g., VTRB phase predict easy)."""
    if isinstance(model.model, str):
        model_name = model.model
    elif hasattr(model, 'model_path'):
        model_name = model.model_path.split('/')[1]
    else:
        model_name = model.name

    write_dir = osp.join(work_dir, task['name'], model_name, name)
    if not osp.exists(write_dir):
        os.makedirs(write_dir)

    with open(osp.join(write_dir, 'prompt.txt'), 'w') as f:
        f.write(prompt_template)

    for frame, label in tqdm(dataset):
        meta = frame['meta']
        out_file = osp.join(write_dir, f'{meta["id"]}.json')
        if osp.exists(out_file) and not kwargs.get('override_outputs', False):
            continue
        if osp.exists(out_file):
            os.remove(out_file)

        choices_text = "\n".join(f"  {k}: {v}" for k, v in meta['choices'].items())
        prompt = prompt_template.replace('{choices}', choices_text)

        try:
            ret = model.generate([frame['path'], prompt])
            if isinstance(ret, int):
                dump({'answer': None, 'raw_response': f'Blocked: {ret}'}, out_file)
                continue

            answer = None
            ret_clean = ret.strip()
            for letter in ['A', 'B', 'C', 'D', 'E']:
                if ret_clean.startswith(letter) or ret_clean == letter:
                    answer = letter
                    break
            if answer is None:
                for letter in ['A', 'B', 'C', 'D', 'E']:
                    if letter in ret_clean[:10]:
                        answer = letter
                        break

            dump({'answer': answer, 'raw_response': ret}, out_file)

        except Exception as e:
            print(f'Exception: {e}')
            dump({'answer': None, 'raw_response': f'Exception: {e}'}, out_file)
            continue

    preds, labels = eval_data_mcq(model, work_dir, name, dataset, task)
    return preds, labels


def eval_data_mcq(
    model,
    work_dir,
    name,
    dataset,
    task,
    *args,
    **kwargs
):
    """Evaluate MCQ predictions."""
    if isinstance(model.model, str):
        model_name = model.model
    elif hasattr(model, 'model_path'):
        model_name = model.model_path.split('/')[1]
    else:
        model_name = model.name

    read_dir = osp.join(work_dir, task['name'], model_name, name)
    task_name = task['name']

    all_preds = []
    all_labels = []

    for idx in range(len(dataset)):
        frame_path, label, meta = dataset.labels[idx]
        out_file = osp.join(read_dir, f'{meta["id"]}.json')

        if not osp.exists(out_file):
            all_preds.append(None)
            all_labels.append(label)
            continue

        with open(out_file, 'r') as f:
            result = json.load(f)

        all_preds.append(result.get('answer', None))
        all_labels.append(label)

    valid = [(p, l) for p, l in zip(all_preds, all_labels) if p is not None]
    if valid:
        preds_v, labels_v = zip(*valid)
        correct = sum(1 for p, l in zip(preds_v, labels_v) if p == l)
        acc = round(correct / len(valid), 3)

        from collections import Counter
        pos_correct = Counter()
        pos_total = Counter()
        for p, l in zip(preds_v, labels_v):
            pos_total[l] += 1
            if p == l:
                pos_correct[l] += 1

        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Valid', 'Total'] + [f'Acc_{l}' for l in sorted(pos_total.keys())],
            'Value': [acc, len(valid), len(all_preds)] +
                     [round(pos_correct[l] / pos_total[l], 3) for l in sorted(pos_total.keys())],
        })
    else:
        metrics_df = pd.DataFrame({'Metric': ['Error'], 'Value': ['No valid predictions']})

    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(osp.join(work_dir, f'metrics_{task_name}_{model_name.replace("/","")}_{name}.csv'), index=False)
    return all_preds, all_labels


# ── MCQ Multi-frame (per-sample choices + multi-frame input) ────────────────

def infer_data_mcq_multiframe(
    model,
    work_dir,
    name,
    dataset,
    task,
    prompt_template,
    **kwargs
):
    """Inference for multi-frame MCQ tasks with per-sample choices."""
    if isinstance(model.model, str):
        model_name = model.model
    elif hasattr(model, 'model_path'):
        model_name = model.model_path.split('/')[1]
    else:
        model_name = model.name

    write_dir = osp.join(work_dir, task['name'], model_name, name)
    if not osp.exists(write_dir):
        os.makedirs(write_dir)

    with open(osp.join(write_dir, 'prompt.txt'), 'w') as f:
        f.write(prompt_template)

    task_name = task['name']

    for frame, label in tqdm(dataset):
        meta = frame['meta']
        out_file = osp.join(write_dir, f'{meta["id"]}.json')
        if osp.exists(out_file) and not kwargs.get('override_outputs', False):
            continue
        if osp.exists(out_file):
            os.remove(out_file)

        # Fill choices into prompt
        choices = meta.get('choices', {})
        choices_text = "\n".join(f"  {k}: {v}" for k, v in choices.items())
        prompt = prompt_template.replace('{choices}', choices_text)

        # Multi-frame message
        if 'paths' in frame:
            message = frame['paths'] + [prompt]
        else:
            message = [frame['path'], prompt]

        try:
            ret = model.generate(message)
            if isinstance(ret, int):
                dump({'answer': None, 'raw_response': f'Blocked: {ret}'}, out_file)
                continue

            # Parse JSON response for current_phase and next_phase (as letters)
            start_index = ret.find('{')
            end_index = ret.rfind('}')
            answer = None
            if start_index != -1 and end_index != -1:
                result = ret[start_index:end_index + 1].replace('\n', '')
                result = result.replace('False', '0').replace('True', '1')
                try:
                    parsed = json.loads(result)
                    answer = {
                        'current_phase': parsed.get('current_phase', ''),
                        'next_phase': parsed.get('next_phase', ''),
                    }
                except json.JSONDecodeError:
                    pass

            dump({'answer': answer, 'raw_response': ret}, out_file)

        except Exception as e:
            print(f'Exception: {e}')
            dump({'answer': None, 'raw_response': f'Exception: {e}'}, out_file)
            continue

    preds, labels = eval_data_mcq_multiframe(model, work_dir, name, dataset, task)
    return preds, labels


def eval_data_mcq_multiframe(
    model,
    work_dir,
    name,
    dataset,
    task,
    *args,
    **kwargs
):
    """Evaluate multi-frame MCQ planning predictions."""
    if isinstance(model.model, str):
        model_name = model.model
    elif hasattr(model, 'model_path'):
        model_name = model.model_path.split('/')[1]
    else:
        model_name = model.name

    read_dir = osp.join(work_dir, task['name'], model_name, name)
    task_name = task['name']

    all_preds = []
    all_metas = []

    for idx in range(len(dataset)):
        frame_paths, label, meta = dataset.labels[idx]
        out_file = osp.join(read_dir, f'{meta["id"]}.json')

        if not osp.exists(out_file):
            all_preds.append(None)
            all_metas.append(meta)
            continue

        with open(out_file, 'r') as f:
            result = json.load(f)

        all_preds.append(result.get('answer', None))
        all_metas.append(meta)

    # Evaluate: map letter answers back to phase names via choices
    valid = [(p, m) for p, m in zip(all_preds, all_metas) if p is not None and isinstance(p, dict)]
    if valid:
        choices_map = lambda m: m.get('choices', {})

        cur_correct = 0
        next_correct = 0
        for p, m in valid:
            ch = choices_map(m)
            pred_cur = ch.get(p.get('current_phase', ''), p.get('current_phase', ''))
            pred_next = ch.get(p.get('next_phase', ''), p.get('next_phase', ''))
            gt_cur = m['phase']
            gt_next = m['next_phase']
            if pred_cur.lower() == gt_cur.lower():
                cur_correct += 1
            if pred_next.lower() == gt_next.lower():
                next_correct += 1

        metrics_df = pd.DataFrame({
            'Metric': ['Current Phase Acc', 'Next Phase Acc', 'Valid', 'Total'],
            'Value': [
                round(cur_correct / len(valid), 3),
                round(next_correct / len(valid), 3),
                len(valid),
                len(all_preds),
            ]
        })
    else:
        metrics_df = pd.DataFrame({'Metric': ['Error'], 'Value': ['No valid predictions']})

    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(osp.join(work_dir, f'metrics_{task_name}_{model_name.replace("/","")}_{name}.csv'), index=False)
    return all_preds, all_metas
