import os.path as osp

def get_prompts(path, task, model):

        PROMPTS = {}

        # DRESDEN ANATOMY
        PROMPTS[('dresden_anatomy_presence', 'GeminiPro1-5')] = 'Which of these anatomical structure is visible in this image: \
                "abdominal wall", "colon", "inferior mesenteric artery", "intestinal veins", "liver", "pancreas", "small intestine", \
                "spleen", "stomach", "ureter", "vesicular glands". Respond with True or False for all structures according to whether \
                or not the anatomy is visible. Use this JSON schema: {"anatomy_name": bool} and avoid line breaks.'

        dresden_anatomy = ["the abdominal wall", "the colon", "the inferior mesenteric artery", "intestinal veins", "the liver", "the pancreas", "the small intestine", "the spleen", "the stomach", "the ureter", "vesicular glands"]
        candidate_captions_positive = ["Answer en Is %s in this image?" % cls for cls in dresden_anatomy]
        PROMPTS[('dresden_anatomy_presence', 'paligemma-3b-mix-448')] = candidate_captions_positive

        candidate_captions_positive = ["A surgical scene containing %s." % cls for cls in dresden_anatomy]
        PROMPTS[('dresden_anatomy_presence', 'CLIP')] =  candidate_captions_positive 

        PROMPTS[('dresden_anatomy_presence', 'SurgVLP')] =  ["I see the abdominal wall", 
                                                     "I see the colon", 
                                                     "I see the inferior mesenteric artery", 
                                                     "I see intestinal veins", 
                                                     "I see the liver", 
                                                     "I see the pancreas", 
                                                     "I see the small intestine", 
                                                     "I see the spleen", 
                                                     "I see the stomach", 
                                                     "I see the ureter", 
                                                     "I see vesicular glands"]
        
        # ENDOSCAPES CVS 
        PROMPTS[('endoscapes_cvs_assessment', 'GeminiPro1-5')] = 'You are a helpful medical video assistant. \
        Task: Assess whether Critical View of Safety (CVS) is fully achieved in the provided frames from a cholecystectomy video. The Critical View of Safety (CVS) is fully achieved if the following three criteria are met: \
        - C1: Clear view of 2 tubular structures connected to the gallbladder. \
        - C2: A carefully dissected hepatocystic triangle presenting an unimpeded view of only the 2 cystic structures and the cystic plate. \
        - C3: The lower third of the gallbladder is dissected off the cystic plate. \
        Instructions: Assess the image carefully, and answer which of the Critical View of Safety (CVS) criteria are met. \
        Use this JSON schema: {"C1": bool, "C2": bool, "C3": bool} and avoid line breaks.'

        PROMPTS[('endoscapes_cvs_assessment', 'GPT4o')] = 'You are a helpful medical video assistant. \
        Task: Assess whether Critical View of Safety (CVS) is fully achieved in the provided frames from a cholecystectomy video. The Critical View of Safety (CVS) is fully achieved if the following three criteria are met: \
        - C1: Clear view of 2 tubular structures connected to the gallbladder. \
        - C2: A carefully dissected hepatocystic triangle presenting an unimpeded view of only the 2 cystic structures and the cystic plate. \
        - C3: The lower third of the gallbladder is dissected off the cystic plate. \
        Instructions: Assess the image carefully, and answer which of the Critical View of Safety (CVS) criteria are met. \
        Use this JSON schema: {"C1": bool, "C2": bool, "C3": bool} and avoid line breaks.'  # GPT gets its own prompt as the Gemini prompt was designed by Med-Gemini authors and was copied from their paper. That prompt does not work well for GPT.
        
        PROMPTS[('endoscapes_cvs_assessment', 'Qwen2-VL-7B-Instruct')] = PROMPTS[('endoscapes_cvs_assessment', 'GPT4o')]  # Qwen2 gets the same prompt as GPT4o
        PROMPTS[('endoscapes_cvs_assessment', 'llava_next_vicuna_7b')] = PROMPTS[('endoscapes_cvs_assessment', 'GPT4o')]
        PROMPTS[('endoscapes_cvs_assessment', 'CLIP')] = ['Clear view of 2 tubular structures connected to the gallbladder', 'A carefully dissected hepatocystic triangle presenting an unimpeded view of only the 2 cystic structures and the cystic plate', 'The lower third of the gallbladder is dissected off the cystic plate.']
        PROMPTS[('endoscapes_cvs_assessment', 'SurgVLP')] = ['2 tubular structures are connected to the gallbladder', 'A carefully dissected hepatocystic triangle is presenting an unimpeded view of only the 2 cystic structures and the cystic plate', 'The lower third of the gallbladder is dissected off the cystic plate.']
        PROMPTS[('endoscapes_cvs_assessment', 'paligemma-3b-mix-448')] = ["Answer en Is there a clear view of 2 tubular structures connected to the gallbladder?", "Answer en Is there a carefully dissected hepatocystic triangle presenting an unimpeded view of only the 2 cystic structures and the cystic plate?", "Answer en Is the lower third of the gallbladder dissected off the cystic plate?"]

        # ENDOSCAPES OBJECT DETECTION (tool + anatomy)
        endoscapes_det_categories = ['cystic_plate', 'calot_triangle', 'cystic_artery', 'cystic_duct', 'gallbladder', 'tool']

        PROMPTS[('endoscapes_object_detection', 'GeminiPro1-5')] = 'Detect and localize all objects in this laparoscopic cholecystectomy image. ' \
        'The possible object categories are: cystic_plate, calot_triangle, cystic_artery, cystic_duct, gallbladder, tool. ' \
        'For each detected object, provide the bounding box as [x1, y1, x2, y2] coordinates (0-1000 normalized). ' \
        'Use this JSON schema: {"category_name": [x1, y1, x2, y2]} for single instances, or {"category_name": [[x1,y1,x2,y2], [x1,y1,x2,y2]]} for multiple instances. ' \
        'Only include categories that are visible. Avoid line breaks.'

        PROMPTS[('endoscapes_object_detection', 'llava_next_vicuna_7b')] = PROMPTS[('endoscapes_object_detection', 'GeminiPro1-5')]

        PROMPTS[('endoscapes_object_detection', 'paligemma-3b-mix-448')] = [
            "detect cystic plate", "detect calot triangle", "detect cystic artery",
            "detect cystic duct", "detect gallbladder", "detect tool"
        ]

        PROMPTS[('endoscapes_cvs_assessment_threeshot', 'GeminiPro1-5')] = [path + '/train/10_17850.jpg',
                                                          'output: {"C1": true, "C2": false, "C3": false}',
                                                          path + '/train/22_41275.jpg',
                                                          'output: {"C1": false, "C2": true, "C3": false}',
                                                          path + '/train/23_28600.jpg',
                                                          'output: {"C1": false, "C2": false, "C3": true}',
                                                          path + '/train/4_36950.jpg',
                                                          'output: {"C1": true, "C2": true, "C3": true}',
                                                          path + '/train/1_29850.jpg',
                                                          'output: {"C1": false, "C2": false, "C3": false}',
                                                          path + '/train/15_30775.jpg',
                                                          'output: {"C1": true, "C2": true, "C3": true}',
                                                           path + '/train/89_26875.jpg',
                                                          'output: {"C1": true, "C2": true, "C3": true}',
                                                          path + '/train/106_53450.jpg',
                                                          'output: {"C1": false, "C2": true, "C3": false}',
                                                            path + '/train/57_27650.jpg',
                                                          'output: {"C1": false, "C2": false, "C3": true}',
                                                            'You just saw some images of a laparoscopic cholecystectomy with corresponding Critical View of Safety annotations. In the next image, assess whether Critical View of Safety (CVS) is fully achieved in the provided frames from a cholecystectomy video. \
                                                                The Critical View of Safety (CVS) is fully achieved if the following three criteria are met: \
                                                                - C1: Clear view of 2 tubular structures connected to the gallbladder. \
                                                                - C2: A carefully dissected hepatocystic triangle presenting an unimpeded view of only the 2 cystic structures and the cystic plate. \
                                                                - C3: The lower third of the gallbladder is dissected off the cystic plate. \
                                                            Instructions: Assess the image carefully, and answer which of the Critical View of Safety (CVS) criteria are met. \
                                                            Use this JSON schema: {"C1": bool, "C2": bool, "C3": bool} and avoid line breaks.']


        PROMPTS[('endoscapes_cvs_assessment_oneshot', 'GeminiPro1-5')] = [path + '/train/10_17850.jpg',
                                                                'output: {"C1": true, "C2": false, "C3": false}',
                                                                path + '/train/22_41275.jpg',
                                                                'output: {"C1": false, "C2": true, "C3": false}',
                                                                path + '/train/23_28600.jpg',
                                                                'output: {"C1": false, "C2": false, "C3": true}',
                                                                'You just saw some images of a laparoscopic cholecystectomy with corresponding Critical View of Safety annotations. In the next image, assess whether Critical View of Safety (CVS) is fully achieved in the provided frames from a cholecystectomy video. \
                                                                        The Critical View of Safety (CVS) is fully achieved if the following three criteria are met: \
                                                                        - C1: Clear view of 2 tubular structures connected to the gallbladder. \
                                                                        - C2: A carefully dissected hepatocystic triangle presenting an unimpeded view of only the 2 cystic structures and the cystic plate. \
                                                                        - C3: The lower third of the gallbladder is dissected off the cystic plate. \
                                                                Instructions: Assess the image carefully, and answer which of the Critical View of Safety (CVS) criteria are met. \
                                                                Use this JSON schema: {"C1": bool, "C2": bool, "C3": bool} and avoid line breaks.']

        PROMPTS[('endoscapes_cvs_assessment_fiveshot', 'GeminiPro1-5')] = [path + '/train/95_28500.jpg', 'output: {"C1": 1, "C2": 0, "C3": 0}', 
                                                                           path + '/train/42_46900.jpg', 'output: {"C1": 1, "C2": 0, "C3": 0}', 
                                                                           path + '/train/82_24600.jpg', 'output: {"C1": 1, "C2": 0, "C3": 0}', 
                                                                           path + '/train/88_33075.jpg', 'output: {"C1": 1, "C2": 1, "C3": 1}', 
                                                                           path + '/train/6_16425.jpg', 'output: {"C1": 0, "C2": 0, "C3": 1}',
                                                                           path + '/train/117_21625.jpg', 'output: {"C1": 0, "C2": 0, "C3": 1}', 
                                                                           path + '/train/76_57225.jpg', 'output: {"C1": 1, "C2": 1, "C3": 1}', 
                                                                           path + '/train/31_49400.jpg', 'output: {"C1": 0, "C2": 0, "C3": 1}', 
                                                                           path + '/train/57_34400.jpg', 'output: {"C1": 0, "C2": 1, "C3": 1}', 
                                                                           path + '/train/41_600.jpg', 'output: {"C1": 1, "C2": 1, "C3": 0}', 
                                                                           path + '/train/4_24325.jpg', 'output: {"C1": 0, "C2": 1, "C3": 0}', 
                                                                'You just saw some images of a laparoscopic cholecystectomy with corresponding Critical View of Safety annotations. In the next image, assess whether Critical View of Safety (CVS) is fully achieved in the provided frames from a cholecystectomy video. \
                                                                        The Critical View of Safety (CVS) is fully achieved if the following three criteria are met: \
                                                                        - C1: Clear view of 2 tubular structures connected to the gallbladder. \
                                                                        - C2: A carefully dissected hepatocystic triangle presenting an unimpeded view of only the 2 cystic structures and the cystic plate. \
                                                                        - C3: The lower third of the gallbladder is dissected off the cystic plate. \
                                                                Instructions: Assess the image carefully, and answer which of the Critical View of Safety (CVS) criteria are met. \
                                                                Use this JSON schema: {"C1": bool, "C2": bool, "C3": bool} and avoid line breaks.']


        # Cholec80 / Heichole Phase
        cholec_phases = ['preparation', 'calot triangle dissection', 'clipping cutting', 'gallbladder dissection',
        'gallbladder packaging', 'cleaning coagulation', 'gallbladder retraction']
        candidate_captions_positive = ["A surgical scene during %s." % cls for cls in cholec_phases]
        PROMPTS[('cholec80_phase_recognition', 'CLIP')] =  candidate_captions_positive
        PROMPTS[('heichole_phase_recognition', 'CLIP')] =  candidate_captions_positive

        PROMPTS[('cholec80_phase_recognition', 'SurgVLP')] = ['In preparation phase I insert trocars to patient abdomen cavity',
        'In calot triangle dissection phase I use grasper to hold gallbladder and use hook to expose the hepatic triangle area and cystic duct and cystic artery',
        'In clip and cut phase I use clipper to clip the cystic duct and artery then use scissor to cut them',
        'In dissection phase I use the hook to dissect the connective tissue between gallbladder and liver' ,
        'In packaging phase I put the gallbladder into the specimen bag' ,
        'In clean and coagulation phase I use suction and irrigation to clear the surgical field and coagulate bleeding vessels',
        'In retraction phase I grasp the specimen bag and remove it from trocar']

        PROMPTS[('heichole_phase_recognition', 'SurgVLP')]  = PROMPTS[('cholec80_phase_recognition', 'SurgVLP')]

        PROMPTS[('cholec80_phase_recognition', 'paligemma-3b-mix-448')] = "Answer en What is the surgical phase shown in this image? Choose from: Preparation, Calot Triangle Dissection, Clipping Cutting, Gallbladder Dissection, Gallbladder Packaging, Cleaning Coagulation, Gallbladder Retraction."
        PROMPTS[('heichole_phase_recognition', 'paligemma-3b-mix-448')] = PROMPTS[('cholec80_phase_recognition', 'paligemma-3b-mix-448')]

        PROMPTS[('cholec80_phase_recognition', 'GeminiPro1-5')] = 'You are shown an image captured during a laparoscopic cholecystectomy. Determine the surgical phase of the image. The possible phases are \
        0: Preparation, 1: Calot Triangle Dissection, 2: Clipping Cutting, 3: Gallbladder Dissection, \
        4: Gallbladder Packaging, 5: Cleaning Coagulation, 6: Gallbladder Retraction. There are no other options. Use this JSON schema: {"phase": int} and avoid line breaks. '
        PROMPTS[('heichole_phase_recognition', 'GeminiPro1-5')] = PROMPTS[('cholec80_phase_recognition', 'GeminiPro1-5')]
        PROMPTS[('cholec80_phase_recognition', 'llava_next_vicuna_7b')] = PROMPTS[('cholec80_phase_recognition', 'GeminiPro1-5')]

        PROMPTS[('heichole_phase_recognition_oneshot', 'GeminiPro1-5')] =[
        path + '/Hei-Chole11/frame_00000.png',  # train split [12, 11, 24]
        'output: {"phase": 0}',
        path + '/Hei-Chole12/frame_12569.png', 
        'output: {"phase": 1}', 
        path + '/Hei-Chole12/frame_09762.png', 
        'output: {"phase": 2}', 
        path + '/Hei-Chole11/frame_20595.png', 
        'output: {"phase": 3}', 
        path + '/Hei-Chole12/frame_12569.png', 
        'output: {"phase": 4}', 
        path + '/Hei-Chole24/frame_116426.png', 
        'output: {"phase": 5}', 
        path + '/Hei-Chole24/frame_123686.png', 
        'output: {"phase": 6}', 
        'You just saw some images of a laparoscopic cholecystectomy and their coressponding phase annotations.  For the next image, determine the surgical phase of the image. The possible phases are \
        0: Preparation, 1: Calot Triangle Dissection, 2: Clipping Cutting, 3: Gallbladder Dissection, \
        4: Gallbladder Packaging, 5: Cleaning Coagulation, 6: Gallbladder Retraction. There are no other options. Use this JSON schema: {"phase": int} and avoid line breaks.']

        PROMPTS[('heichole_phase_recognition_threeshot', 'GeminiPro1-5')] =[
                path + '/Hei-Chole11/frame_18950.png', 'output: {"phase": 3}', 
                path + '/Hei-Chole24/frame_121100.png', 'output: {"phase": 6}', 
                path + '/Hei-Chole12/frame_25575.png', 'output: {"phase": 2}', 
                path + '/Hei-Chole24/frame_36700.png', 'output: {"phase": 1}', 
                path + '/Hei-Chole24/frame_40025.png', 'output: {"phase": 1}', 
                path + '/Hei-Chole24/frame_124450.png', 'output: {"phase": 6}', 
                path + '/Hei-Chole24/frame_50325.png', 'output: {"phase": 1}',
                path + '/Hei-Chole24/frame_00900.png', 'output: {"phase": 0}', 
                path + '/Hei-Chole11/frame_25125.png', 'output: {"phase": 5}', 
                path + '/Hei-Chole24/frame_128050.png', 'output: {"phase": 5}', 
                path + '/Hei-Chole11/frame_34925.png', 'output: {"phase": 5}', 
                path + '/Hei-Chole24/frame_89050.png', 'output: {"phase": 3}', 
                path + '/Hei-Chole11/frame_12500.png', 'output: {"phase": 2}', 
                path + '/Hei-Chole12/frame_01550.png', 'output: {"phase": 0}', 
                path + '/Hei-Chole12/frame_34550.png', 'output: {"phase": 4}', 
                path + '/Hei-Chole24/frame_123600.png', 'output: {"phase": 6}', 
                path + '/Hei-Chole24/frame_110625.png', 'output: {"phase": 4}', 
                path + '/Hei-Chole24/frame_03925.png', 'output: {"phase": 0}', 
                path + '/Hei-Chole24/frame_93400.png', 'output: {"phase": 3}', 
                path + '/Hei-Chole24/frame_100025.png', 'output: {"phase": 2}', 
                path + '/Hei-Chole24/frame_105775.png', 'output: {"phase": 4}'
        'You just saw some images of a laparoscopic cholecystectomy and their coressponding phase annotations.  For the next image, determine the surgical phase of the image. The possible phases are \
        0: Preparation, 1: Calot Triangle Dissection, 2: Clipping Cutting, 3: Gallbladder Dissection, \
        4: Gallbladder Packaging, 5: Cleaning Coagulation, 6: Gallbladder Retraction. There are no other options. Use this JSON schema: {"phase": int} and avoid line breaks.']

        PROMPTS[('heichole_phase_recognition_fiveshot', 'GeminiPro1-5')] =[
                path + '/Hei-Chole24/frame_124700.png', 'output: {"phase": 6}', 
                path + '/Hei-Chole24/frame_22650.png', 'output: {"phase": 1}', 
                path + '/Hei-Chole24/frame_106975.png', 'output: {"phase": 4}', 
                path + '/Hei-Chole24/frame_33650.png', 'output: {"phase": 1}', 
                path + '/Hei-Chole24/frame_106150.png', 'output: {"phase": 4}', 
                path + '/Hei-Chole11/frame_01275.png', 'output: {"phase": 0}', 
                path + '/Hei-Chole24/frame_39275.png', 'output: {"phase": 1}', 
                path + '/Hei-Chole12/frame_02825.png', 'output: {"phase": 1}', 
                path + '/Hei-Chole24/frame_110675.png', 'output: {"phase": 4}', 
                path + '/Hei-Chole24/frame_20275.png', 'output: {"phase": 1}', 
                path + '/Hei-Chole12/frame_30325.png', 'output: {"phase": 3}', 
                path + '/Hei-Chole24/frame_116700.png', 'output: {"phase": 5}', 
                path + '/Hei-Chole24/frame_118050.png', 'output: {"phase": 5}', 
                path + '/Hei-Chole24/frame_108100.png', 'output: {"phase": 4}', 
                path + '/Hei-Chole12/frame_24300.png', 'output: {"phase": 2}', 
                path + '/Hei-Chole24/frame_91300.png', 'output: {"phase": 3}', 
                path + '/Hei-Chole24/frame_109075.png', 'output: {"phase": 4}', 
                path + '/Hei-Chole24/frame_02575.png', 'output: {"phase": 0}', 
                path + '/Hei-Chole24/frame_103275.png', 'output: {"phase": 2}', 
                path + '/Hei-Chole24/frame_94250.png', 'output: {"phase": 3}', 
                path + '/Hei-Chole12/frame_29925.png', 'output: {"phase": 3}', 
                path + '/Hei-Chole11/frame_35175.png', 'output: {"phase": 5}', 
                path + '/Hei-Chole12/frame_43725.png', 'output: {"phase": 5}', 
                path + '/Hei-Chole12/frame_30825.png', 'output: {"phase": 3}', 
                path + '/Hei-Chole24/frame_100550.png', 'output: {"phase": 2}', 
                path + '/Hei-Chole12/frame_24575.png', 'output: {"phase": 2}', 
                path + '/Hei-Chole24/frame_124875.png', 'output: {"phase": 6}', 
                path + '/Hei-Chole12/frame_37700.png', 'output: {"phase": 5}', 
                path + '/Hei-Chole11/frame_14175.png', 'output: {"phase": 2}', 
                path + '/Hei-Chole11/frame_02600.png', 'output: {"phase": 0}', 
                path + '/Hei-Chole24/frame_125500.png', 'output: {"phase": 6}', 
                path + '/Hei-Chole24/frame_122425.png', 'output: {"phase": 6}', 
                path + '/Hei-Chole24/frame_126325.png', 'output: {"phase": 6}', 
                path + '/Hei-Chole24/frame_05225.png', 'output: {"phase": 0}', 
                path + '/Hei-Chole24/frame_03350.png', 'output: {"phase": 0}', 
        'You just saw some images of a laparoscopic cholecystectomy and their coressponding phase annotations.  For the next image, determine the surgical phase of the image. The possible phases are \
        0: Preparation, 1: Calot Triangle Dissection, 2: Clipping Cutting, 3: Gallbladder Dissection, \
        4: Gallbladder Packaging, 5: Cleaning Coagulation, 6: Gallbladder Retraction. There are no other options. Use this JSON schema: {"phase": int} and avoid line breaks.']


        # Cholec 80 Tool 
        cholec_tools = ['grasper', 'bipolar', 'hook', 'scissors', 'clipper', 'irrigator', 'specimen bag']
        candidate_captions_positive = ["A surgical scene containing a %s." % cls for cls in cholec_tools]
        PROMPTS[('cholec80_tool_recognition', 'CLIP')] =  candidate_captions_positive

        PROMPTS[('cholec80_tool_recognition', 'SurgVLP')] = ['I use grasper or cautery forcep to grasp it',
        'I use bipolar to coagulate and clean the bleeding',
        'I use hook to dissect it',
        'I use scissor',
        'I use clipper to clip it',
        'I use irrigator to suck it',
        'I use specimenbag to wrap it']

        PROMPTS[('cholec80_tool_recognition', 'paligemma-3b-mix-448')] = ["Answer en Is there a grasper in this image?", "Answer en Is there a bipolar in this image?", "Answer en Is there a hook in this image?", "Answer en Are there scissors in this image?", "Answer en Is there a clipper in this image?", "Answer en Is there a irrigator in this image?", "Answer en Is there a specimen bag in this image?"]

        PROMPTS[('cholec80_tool_recognition', 'GeminiPro1-5')] = 'Which of these tools is present in the image: Grasper, Bipolar, Hook, Scissors, Clipper, \
        Irrigator, SpecimenBag?\
        Respond with a 0 or 1 for all tools according to whether or not the tool is present. \
        Use this JSON schema: {"Grasper": bool, "Bipolar": bool, "Hook": bool, "Scissors": bool, "Clipper": bool, "Irrigator": bool, "SpecimenBag": bool} and avoid line breaks.'
        PROMPTS[('cholec80_tool_recognition', 'llava_next_vicuna_7b')] = PROMPTS[('cholec80_tool_recognition', 'GeminiPro1-5')]


        # Heichole Tool
        PROMPTS[('heichole_tool_recognition', 'GeminiPro1-5')] = 'Which of these tools is present in the image: Grasper, Bipolar Forceps, Hook, Scissors, Clipper, Irrigator, Specimen Bag? \
        Respond with a 0 or 1 for all tools according to whether or not the tool is present. \
        Use this JSON schema: {"tool_name": bool} and avoid line breaks.'


        heichole_tools = ['a grasper', 'bipolar forceps', 'a hook', 'scissors', 'a clipper', 'an irrigator', 'a specimen bag']
        candidate_captions_positive = ["A surgical scene containing %s." % cls for cls in heichole_tools]
        PROMPTS[('heichole_tool_recognition', 'CLIP')] =  candidate_captions_positive 

        PROMPTS[('heichole_tool_recognition', 'SurgVLP')] = ['I use grasper or cautery forcep to grasp it',
                                                    'I use bipolar to coagulate and clean the bleeding',
                                                    'I use hook to dissect tissue',
                                                    'I use scissor',
                                                    'I use clipper to clip it',
                                                    'I use irrigator to suck it',
                                                    'I use specimenbag to wrap it']
        
        PROMPTS[('heichole_tool_recognition', 'paligemma-3b-mix-448')] = ["Answer en Is there a grasper in this image?", "Answer en Is there a bipolar forceps in this image?", "Answer en Is there a hook in this image?", "Answer en Are there scissors in this image?", "Answer en Is there a clipper in this image?", "Answer en Is there an irrigator in this image?", "Answer en Is there a specimen bag in this image?"]


        PROMPTS[('heichole_tool_recognition_oneshot', 'GeminiPro1-5')] = [
        path + '/Hei-Chole12/frame_09234.png', 
        'output: {"Grasper": false, "Bipolar Forceps": false, "Hook": 0, "Scissors": false, "Clipper": true, "Irrigator": false, "Specimen Bag": false}', 
        path + '/Hei-Chole11/frame_08049.png', 
        'output: {"Grasper": false, "Bipolar Forceps": true, "Hook": 0, "Scissors": false, "Clipper": false, "Irrigator": false, "Specimen Bag": false}',
        path + '/Hei-Chole11/frame_10931.png', 
        'output: {"Grasper": false, "Bipolar Forceps": false, "Hook": 0, "Scissors": true, "Clipper": false, "Irrigator": false, "Specimen Bag": false}', 
        path + '/Hei-Chole11/frame_21704.png', 
        'output: {"Grasper": true, "Bipolar Forceps": false, "Hook": 0, "Scissors": false, "Clipper": false, "Irrigator": true, "Specimen Bag": false}', 
        path + '/Hei-Chole11/frame_31351.png', 
        'output: {"Grasper": false, "Bipolar Forceps": false, "Hook": 0, "Scissors": false, "Clipper": false, "Irrigator": false, "Specimen Bag": true}', 
        path + '/Hei-Chole24/frame_95118.png', 
        'output: {"Grasper": false, "Bipolar Forceps": false, "Hook": 0, "Scissors": false, "Clipper": false, "Irrigator": false, "Specimen Bag": false}', 
        path + '/Hei-Chole12/frame_00003.png', 
        'output: {"Grasper": false, "Bipolar Forceps": false, "Hook": 0, "Scissors": false, "Clipper": false, "Irrigator": false, "Specimen Bag": false}', 
        'You just saw some images of a laparoscopic cholecystectomy with corresponding tool annotations. In the next image, assess which of these tools is present: Grasper, Bipolar Forceps, Hook, Scissors, Clipper, Irrigator, Specimen Bag. \
        Respond with a 0 or 1 for all tools according to whether or not the tool is present. \
        Use this JSON schema: {"tool_name": bool} and avoid line breaks.']

        PROMPTS[('heichole_tool_recognition_threeshot', 'GeminiPro1-5')] = [
                path + '/Hei-Chole24/frame_74625.png', 'output: {"Grasper": 1, "Bipolar Forceps": 1, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}', 
                path + '/Hei-Chole24/frame_12375.png', 'output: {"Grasper": 1, "Bipolar Forceps": 1, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}',
                  path + '/Hei-Chole24/frame_11625.png', 'output: {"Grasper": 1, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}', 
                  path + '/Hei-Chole12/frame_12000.png', 'output: {"Grasper": 0, "Bipolar Forceps": 1, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}', 
                  path + '/Hei-Chole12/frame_40500.png', 'output: {"Grasper": 1, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 1, "Specimen Bag": 0}', 
                  path + '/Hei-Chole24/frame_48375.png', 'output: {"Grasper": 1, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 1, "Specimen Bag": 0}', 
                  path + '/Hei-Chole24/frame_109125.png', 'output: {"Grasper": 1, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 1, "Specimen Bag": 1}', 
                  path + '/Hei-Chole12/frame_42750.png', 'output: {"Grasper": 0, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 1}', 
                  path + '/Hei-Chole24/frame_96000.png', 'output: {"Grasper": 0, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}', 
                  path + '/Hei-Chole12/frame_43125.png', 'output: {"Grasper": 1, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 1}', 
                  path + '/Hei-Chole24/frame_100500.png', 'output: {"Grasper": 1, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 1, "Irrigator": 0, "Specimen Bag": 0}', 
                  path + '/Hei-Chole24/frame_102375.png', 'output: {"Grasper": 1, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 1, "Irrigator": 0, "Specimen Bag": 0}', 
                  path + '/Hei-Chole12/frame_09375.png', 'output: {"Grasper": 0, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 1, "Irrigator": 0, "Specimen Bag": 0}', 
                  path + '/Hei-Chole24/frame_95625.png', 'output: {"Grasper": 0, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}', 
                  path + '/Hei-Chole11/frame_14625.png', 'output: {"Grasper": 0, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 1, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}', 
                  path + '/Hei-Chole24/frame_67875.png', 'output: {"Grasper": 1, "Bipolar Forceps": 1, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}', 
                  path + '/Hei-Chole24/frame_97500.png', 'output: {"Grasper": 0, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}', 
                  path + '/Hei-Chole12/frame_10500.png', 'output: {"Grasper": 1, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 1, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}', 
                  path + '/Hei-Chole24/frame_93000.png', 'output: {"Grasper": 1, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}', 
                  path + '/Hei-Chole12/frame_25875.png', 'output: {"Grasper": 0, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 1, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}', 
                  path + '/Hei-Chole24/frame_74250.png', 'output: {"Grasper": 1, "Bipolar Forceps": 1, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}',
                'You just saw some images of a laparoscopic cholecystectomy with corresponding tool annotations. In the next image, assess which of these tools is present: Grasper, Bipolar Forceps, Hook, Scissors, Clipper, Irrigator, Specimen Bag. \
                Respond with a 0 or 1 for all tools according to whether or not the tool is present. \
                Use this JSON schema: {"tool_name": bool} and avoid line breaks.']

        PROMPTS[('heichole_tool_recognition_fiveshot', 'GeminiPro1-5')] = [
                path + '/Hei-Chole24/frame_48750.png', 'output: {"Grasper": 1, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 1, "Specimen Bag": 0}', 
                path + '/Hei-Chole11/frame_07125.png', 'output: {"Grasper": 1, "Bipolar Forceps": 1, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}', 
                path + '/Hei-Chole24/frame_45000.png', 'output: {"Grasper": 1, "Bipolar Forceps": 1, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}', 
                path + '/Hei-Chole24/frame_77625.png', 'output: {"Grasper": 1, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}', 
                path + '/Hei-Chole12/frame_37125.png', 'output: {"Grasper": 1, "Bipolar Forceps": 1, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}', 
                path + '/Hei-Chole24/frame_76500.png', 'output: {"Grasper": 1, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 1, "Irrigator": 0, "Specimen Bag": 0}', 
                path + '/Hei-Chole24/frame_49500.png', 'output: {"Grasper": 1, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 1, "Specimen Bag": 0}', 
                path + '/Hei-Chole12/frame_42375.png', 'output: {"Grasper": 0, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 1}', 
                path + '/Hei-Chole12/frame_35250.png', 'output: {"Grasper": 1, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 1}', 
                path + '/Hei-Chole11/frame_18750.png', 'output: {"Grasper": 0, "Bipolar Forceps": 1, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}', 
                path + '/Hei-Chole24/frame_110250.png', 'output: {"Grasper": 1, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 1, "Specimen Bag": 1}', 
                path + '/Hei-Chole24/frame_46125.png', 'output: {"Grasper": 1, "Bipolar Forceps": 1, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}', 
                path + '/Hei-Chole11/frame_09375.png', 'output: {"Grasper": 0, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 1, "Irrigator": 0, "Specimen Bag": 0}', 
                path + '/Hei-Chole12/frame_40875.png', 'output: {"Grasper": 1, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 1, "Specimen Bag": 0}', 
                path + '/Hei-Chole24/frame_102375.png', 'output: {"Grasper": 1, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 1, "Irrigator": 0, "Specimen Bag": 0}', 
                path + '/Hei-Chole24/frame_107625.png', 'output: {"Grasper": 1, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 1}', 
                path + '/Hei-Chole24/frame_124500.png', 'output: {"Grasper": 0, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 1}', 
                path + '/Hei-Chole11/frame_14625.png', 'output: {"Grasper": 0, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 1, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}', 
                path + '/Hei-Chole12/frame_41625.png', 'output: {"Grasper": 1, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 1, "Specimen Bag": 0}', 
                path + '/Hei-Chole12/frame_22500.png', 'output: {"Grasper": 1, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 1, "Irrigator": 0, "Specimen Bag": 0}', 
                path + '/Hei-Chole24/frame_67875.png', 'output: {"Grasper": 1, "Bipolar Forceps": 1, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}', 
                path + '/Hei-Chole12/frame_09000.png', 'output: {"Grasper": 0, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 1, "Irrigator": 0, "Specimen Bag": 0}', 
                path + '/Hei-Chole24/frame_92625.png', 'output: {"Grasper": 1, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}', 
                path + '/Hei-Chole24/frame_96000.png', 'output: {"Grasper": 0, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}', 
                path + '/Hei-Chole11/frame_10875.png', 'output: {"Grasper": 0, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 1, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}', 
                path + '/Hei-Chole24/frame_96750.png', 'output: {"Grasper": 0, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}', 
                path + '/Hei-Chole12/frame_25875.png', 'output: {"Grasper": 0, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 1, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}', 
                path + '/Hei-Chole12/frame_10125.png', 'output: {"Grasper": 0, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 1, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}', 
                path + '/Hei-Chole24/frame_93000.png', 'output: {"Grasper": 1, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}', 
                path + '/Hei-Chole24/frame_95250.png', 'output: {"Grasper": 1, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}', 
                path + '/Hei-Chole24/frame_92250.png', 'output: {"Grasper": 1, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}', 
                path + '/Hei-Chole24/frame_74250.png', 'output: {"Grasper": 1, "Bipolar Forceps": 1, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}', 
                path + '/Hei-Chole24/frame_95625.png', 'output: {"Grasper": 0, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}', 
                path + '/Hei-Chole12/frame_10875.png', 'output: {"Grasper": 0, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 1, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}', 
                path + '/Hei-Chole24/frame_97500.png', 'output: {"Grasper": 0, "Bipolar Forceps": 0, "Hook": 0, "Scissors": 0, "Clipper": 0, "Irrigator": 0, "Specimen Bag": 0}',
                'You just saw some images of a laparoscopic cholecystectomy with corresponding tool annotations. In the next image, assess which of these tools is present: Grasper, Bipolar Forceps, Hook, Scissors, Clipper, Irrigator, Specimen Bag. \
        Respond with a 0 or 1 for all tools according to whether or not the tool is present. \
        Use this JSON schema: {"tool_name": bool} and avoid line breaks.']


        # HeiChole Action
        PROMPTS[('heichole_action_recognition', 'paligemma-3b-mix-448')]  = ["Answer en Does one of the depicted tools grasp something?", "Answer en Does one of the depicted tools hold something?",
                                                                    "Answer en Does one of the depicted tools perform cutting on something?", "Answer en Does one of the depicted tools clip something?"] # Note: PaliGemma doesn't allow "cut" so using "cutting"

        PROMPTS[('heichole_action_recognition', 'llava_next_vicuna_7b')]  = 'For each action in (grasp, hold, cut, clip) return a boolean indicating if the action is performed by any instrument in the image. \
        Use this JSON schema: {"grasp": bool, "hold": bool, "cut": bool, "clip": bool} and avoid line breaks'

        PROMPTS[('heichole_action_recognition', 'GeminiPro1-5')] = 'You are shown an image captured during a laparoscopic cholecystectomy. Find all tools. For each tool, decide if the tool performs one of the following actions: \
        grasp, hold, cut, or clip. It is possible that the insturment is idle and no action is performed. Aggregate the actions across all instruments. For each action return a boolean indicating if the action is performed by any instrument. \
        Use this JSON schema: {"action": bool} and avoid line breaks. An example output could look like this: {"grasp": bool, "hold": bool, "cut": bool, "clip": bool}.'

        PROMPTS[('heichole_action_recognition', 'CLIP')] = ['A surgical tool grasping something', #grasp, hold, cut, or clip.
                                                    'A surgical tool holding something',
                                                    'A surgical tool cutting something',
                                                    'A surgical tool clipping something']

        PROMPTS[('heichole_action_recognition', 'SurgVLP')] = ['I grasp it', #grasp, hold, cut, or clip.
                                                    'I hold it',
                                                    'I cut it',
                                                    'I clip it']
        
        PROMPTS[('heichole_action_recognition_oneshot', 'GeminiPro1-5')] = [
        path + '/Hei-Chole12/frame_09234.png', 
        'output: {"grasp": false, "hold": true, "cut": false, "clip": true}', 
        path + '/Hei-Chole11/frame_10931.png', 
        'output: {"grasp": false, "hold": true, "cut": true, "clip": false}', 
        path + '/Hei-Chole12/frame_02987.png', 
        'output: {"grasp": true, "hold": false, "cut": false, "clip": false}', 
        path + '/Hei-Chole11/frame_14298.png', 
        'output: {"grasp": false, "hold": true, "cut": false, "clip": true}', 
        'You just saw some images of a laparoscopic cholecystectomy with corresponding action annotations. In the next image, find all tools. For each tool, decide which action it performs: \
        grasp, hold, cut, or clip. It is possible that the insturment is idle and no action is performed. Aggregate the actions across all instruments. For each action return a boolean indicating if the action is performed by any instrument. \
        Use this JSON schema: {"grasp": bool, "hold": bool, "cut": bool, "clip": bool} and avoid line breaks. ']

        PROMPTS[('heichole_action_recognition_threeshot', 'GeminiPro1-5')] = [
                path + '/Hei-Chole24/frame_112875.png', 
                'output: {"grasp": 0, "hold": 1, "cut": 0, "clip": 0}', 
                path + '/Hei-Chole24/frame_79500.png', 
                'output: {"grasp": 0, "hold": 1, "cut": 0, "clip": 0}', 
                path + '/Hei-Chole24/frame_84750.png', 
                'output: {"grasp": 0, "hold": 1, "cut": 0, "clip": 0}', 
                path + '/Hei-Chole24/frame_117375.png', 
                'output: {"grasp": 1, "hold": 1, "cut": 0, "clip": 0}', 
                path + '/Hei-Chole24/frame_109500.png', 
                'output: {"grasp": 1, "hold": 1, "cut": 0, "clip": 0}', 
                path + '/Hei-Chole11/frame_13875.png', 
                'output: {"grasp": 0, "hold": 1, "cut": 0, "clip": 1}', 
                path + 'Hei-Chole11/frame_26625.png', 
                'output: {"grasp": 0, "hold": 1, "cut": 0, "clip": 1}', 
                path + 'Hei-Chole12/frame_44250.png', 
                'output: {"grasp": 1, "hold": 0, "cut": 0, "clip": 0}', 
                path + 'Hei-Chole11/frame_14250.png', 
                'output: {"grasp": 0, "hold": 1, "cut": 0, "clip": 1}',
                'You just saw some images of a laparoscopic cholecystectomy with corresponding action annotations. In the next image, find all tools. For each tool, decide which action it performs: \
        grasp, hold, cut, or clip. It is possible that the insturment is idle and no action is performed. Aggregate the actions across all instruments. For each action return a boolean indicating if the action is performed by any instrument. \
        Use this JSON schema: {"grasp": bool, "hold": bool, "cut": bool, "clip": bool} and avoid line breaks. ']

        PROMPTS[('heichole_action_recognition_fiveshot', 'GeminiPro1-5')] = [
                path + 'Hei-Chole24/frame_43125.png', 
                'output: {"grasp": 0, "hold": 1, "cut": 0, "clip": 0}', 
                path + 'Hei-Chole24/frame_43875.png', 
                'output: {"grasp": 0, "hold": 1, "cut": 0, "clip": 0}', 
                path + 'Hei-Chole24/frame_20250.png', 
                'output: {"grasp": 0, "hold": 1, "cut": 0, "clip": 0}', 
                path + 'Hei-Chole24/frame_109875.png', 
                'output: {"grasp": 0, "hold": 1, "cut": 0, "clip": 0}', 
                path + 'Hei-Chole24/frame_123750.png', 
                'output: {"grasp": 0, "hold": 1, "cut": 0, "clip": 0}', 
                path + 'Hei-Chole12/frame_44250.png', 
                'output: {"grasp": 1, "hold": 0, "cut": 0, "clip": 0}', 
                path + 'Hei-Chole24/frame_117375.png', 
                'output: {"grasp": 1, "hold": 1, "cut": 0, "clip": 0}', 
                path + 'Hei-Chole11/frame_13875.png', 
                'output: {"grasp": 0, "hold": 1, "cut": 0, "clip": 1}', 
                path + 'Hei-Chole12/frame_03375.png', 
                'output: {"grasp": 1, "hold": 1, "cut": 0, "clip": 0}', 
                path + 'Hei-Chole11/frame_26625.png', 
                'output: {"grasp": 0, "hold": 1, "cut": 0, "clip": 1}', 
                path + 'Hei-Chole24/frame_34875.png', 
                'output: {"grasp": 1, "hold": 1, "cut": 0, "clip": 0}', 
                path + 'Hei-Chole11/frame_10125.png', 
                'output: {"grasp": 0, "hold": 1, "cut": 0, "clip": 1}', 
                path + 'Hei-Chole24/frame_09375.png', 
                'output: {"grasp": 1, "hold": 0, "cut": 0, "clip": 0}', 
                path + 'Hei-Chole11/frame_14250.png', 
                'output: {"grasp": 0, "hold": 1, "cut": 0, "clip": 1}',
                'You just saw some images of a laparoscopic cholecystectomy with corresponding action annotations. In the next image, find all tools. For each tool, decide which action it performs: \
        grasp, hold, cut, or clip. It is possible that the insturment is idle and no action is performed. Aggregate the actions across all instruments. For each action return a boolean indicating if the action is performed by any instrument. \
        Use this JSON schema: {"grasp": bool, "hold": bool, "cut": bool, "clip": bool} and avoid line breaks. ']


        # AVOS action recognition
        PROMPTS[('avos_action_recognition', 'GeminiPro1-5')] = 'You are shown an image captured during an open surgery. Determine the action being performed in the image. The possible actions are \
        0: cutting - if the surgeon is using a tool like scissors, scalpel, knife, or electrocautery device to cut or dissect tissues. 1: tying - if the surgeon is using their hands or needle holders to create secure knots. 2: suturing - if the surgeon is closing an open wound with a needle but not creating secure knots. 3: background - if no surgical action is being performed, including actions like using forceps, clamps, retractors, or dilators. There are no other options. Use this JSON schema: {"action": int} and avoid line breaks. '


        PROMPTS[('avos_action_recognition_oneshot', 'GeminiPro1-5')] = [path + 'EUdac6A9n60-000006767.jpg',
                                                                        'output: {"action": 3}',
                                                                        path + 'Xg6vD3vngLQ-000000975.jpg',
                                                                        'output: {"action": 0}',
                                                                        path + 'FUGhWj5iv70-000008279.jpg',
                                                                        'output: {"action": 2}',
                                                                        path + 'JUTyS7ZRRkQ-000006353.jpg',
                                                                        'output: {"action": 1}',
                                                                        'You are shown an image captured during an open surgery. Determine the action being performed in the image. The possible actions are \
                                                                        0: cutting - if the surgeon is using a tool like scissors, scalpel, knife, or electrocautery device to cut or dissect tissues. \
                                                                        1: tying - if the surgeon is using their hands or needle holders to create secure knots. \
                                                                                2: suturing - if the surgeon is closing an open wound with a needle but not creating secure knots. \
                                                                                3: background - if no surgical action is being performed, including actions like using forceps, clamps, retractors, or dilators. \
                                                                                        There are no other options. Use this JSON schema: {"action": int} and avoid line breaks. '
                                                                        ]

        PROMPTS[('avos_action_recognition_threeshot', 'GeminiPro1-5')] = [path + 'S4p9MmTfVTs-000001445.jpg', 
                                                                  'output: {"action": 3}', 
                                                                  path + 'RWHBTwfa5C8-000003575.jpg', 
                                                                  'output: {"action": 3}', 
                                                                  path + 'toE_4MtsqQM-000000488.jpg', 
                                                                  'output: {"action": 0}', 
                                                                  path + 'dPvRrcSsc6Y-000000662.jpg', 
                                                                  'output: {"action": 3}', 
                                                                  path + 'cpgMJ7KOVl8-000004213.jpg', 
                                                                  'output: {"action": 0}', 
                                                                  path + 'N32N6VEcW2I-000007967.jpg', 
                                                                  'output: {"action": 2}', 
                                                                  path + 'vtK1XN4ZaU4-000003552.jpg', 
                                                                  'output: {"action": 0}', 
                                                                  path + 'TFwFMav_cpE-000014219.jpg', 
                                                                  'output: {"action": 1}', 
                                                                  path + 'N32N6VEcW2I-000007303.jpg', 
                                                                  'output: {"action": 1}', 
                                                                  path + 'VtJtGtC3R80-000003911.jpg', 
                                                                  'output: {"action": 1}', 
                                                                  path + 'SNsUtI82de8-000009903.jpg', 
                                                                  'output: {"action": 2}', 
                                                                  path + 'e12tIDPDfwU-000003095.jpg', 
                                                                  'output: {"action": 2}',
                                                                        'You are shown an image captured during an open surgery. Determine the action being performed in the image. The possible actions are \
                                                                        0: cutting - if the surgeon is using a tool like scissors, scalpel, knife, or electrocautery device to cut or dissect tissues. \
                                                                        1: tying - if the surgeon is using their hands or needle holders to create secure knots. \
                                                                                2: suturing - if the surgeon is closing an open wound with a needle but not creating secure knots. \
                                                                                3: background - if no surgical action is being performed, including actions like using forceps, clamps, retractors, or dilators. \
                                                                                        There are no other options. Use this JSON schema: {"action": int} and avoid line breaks. ']

        PROMPTS[('avos_action_recognition_fiveshot', 'GeminiPro1-5')] = [path + 'EswP8VDC85s-000000799.jpg', 
                                                                        'output: {"action": 0}', 
                                                                        path + 'synW6molzgA-000005228.jpg', 
                                                                        'output: {"action": 0}', 
                                                                        path + 'l5h_tOU_D9w-000000254.jpg', 
                                                                        'output: {"action": 0}', 
                                                                        path + 'GJ5RwKonnms-000001211.jpg', 
                                                                        'output: {"action": 0}', 
                                                                        path + 'L8k75Onag_o-000006575.jpg', 
                                                                        'output: {"action": 0}', 
                                                                        path + 'oD5gC2ESBnk-000006759.jpg', 
                                                                        'output: {"action": 3}', 
                                                                        path + 'FotC4hB7Y0c-000002397.jpg', 
                                                                        'output: {"action": 2}', 
                                                                        path + 'ou4iO5ah9ys-000000995.jpg', 
                                                                        'output: {"action": 3}', 
                                                                        path + 'ytgWAMS1SkE-000007629.jpg', 
                                                                        'output: {"action": 3}', 
                                                                        path + 'V7vkRKaUkn8-000008998.jpg', 
                                                                        'output: {"action": 3}', 
                                                                        path + 'DpeAsOXVruw-000003135.jpg', 
                                                                        'output: {"action": 2}', 
                                                                        path + '6idNh90AdtA-000001685.jpg', 
                                                                        'output: {"action": 3}', 
                                                                        path + 'VtJtGtC3R80-000007171.jpg', 
                                                                        'output: {"action": 1}', 
                                                                        path + 'S1R95eOuSNk-000002039.jpg', 
                                                                        'output: {"action": 1}', 
                                                                        path + 'S1R95eOuSNk-000004079.jpg', 
                                                                        'output: {"action": 1}', 
                                                                        path + 'S4p9MmTfVTs-000004337.jpg', 
                                                                        'output: {"action": 1}', 
                                                                        path + '3Ql0fGVrQeA-000003717.jpg', 
                                                                        'output: {"action": 2}', 
                                                                        path + 'ytgWAMS1SkE-000009809.jpg', 
                                                                        'output: {"action": 2}', 
                                                                        path + 'LgmXCOICHLA-000002384.jpg', 
                                                                        'output: {"action": 1}', 
                                                                        path + 'GwHruH8trhg-000001674.jpg', 
                                                                        'output: {"action": 2}',
                                                                        'You are shown an image captured during an open surgery. Determine the action being performed in the image. The possible actions are \
                                                                        0: cutting - if the surgeon is using a tool like scissors, scalpel, knife, or electrocautery device to cut or dissect tissues. \
                                                                        1: tying - if the surgeon is using their hands or needle holders to create secure knots. \
                                                                                2: suturing - if the surgeon is closing an open wound with a needle but not creating secure knots. \
                                                                                3: background - if no surgical action is being performed, including actions like using forceps, clamps, retractors, or dilators. \
                                                                                        There are no other options. Use this JSON schema: {"action": int} and avoid line breaks. '
                                                                        ]


        PROMPTS[('avos_action_recognition', 'CLIP')] = ['A surgeon is cutting tissue with scissors, scalpel, knife, or electrocautery device', 'A surgeon is tying knots with their hands or needle holders', 
                                                'A surgeon is suturing an open wound creating straight stitches', 'A surgeon performs a background task like using forceps, clamps, retractors, or dilators']
                                        
        PROMPTS[('avos_action_recognition', 'SurgVLP')] = ['I cut it', #cutting, tying knots, suturing, background task
                                                    'I tie it',
                                                    'I suture it',
                                                    'I perform a background task']
        
        PROMPTS[('avos_action_recognition', 'paligemma-3b-mix-448')] = "Answer en What is the surgical action being performed in this image? Choose from: cutting, tying knots, suturing, background task."

        # Heichole Error Classification
        PROMPTS[('heichole_error_recognition', 'Phi-3.5-Vision')] = 'You are a helpful medical video assistant. You will be provided with separate frames uniformaly sampled from a video segment. \
        Task: classify the surgical error in the video segment. Below are the defined errors: \
        1. Bleeding \
        2. Bile spillage \
        3. Thermal injury \
        4. Perforation \
        Instructions: Assess the images carefully and classify the error. The segment only contains one error. Only output the error in a JSON format, eg: {"error_type": 1}.'
        PROMPTS[('heichole_error_recognition', 'InternVL2-8B')] = PROMPTS[('heichole_error_recognition', 'Phi-3.5-Vision')]
        PROMPTS[('heichole_error_recognition', 'Qwen2-VL-7B-Instruct')] = PROMPTS[('heichole_error_recognition', 'Phi-3.5-Vision')]
        PROMPTS[('heichole_error_recognition', 'GeminiPro1-5')] = 'You are a helpful medical video assistant. \
        Task: Classify which type of error occurs in the provided frames from a cholecystectomy video. \
        The errors include: \
        - 1. Bleeding is defined as blood flowing/moving from a source of injury that is clearly visible on the screen.\
        - 2. Bile spillage is defined as bile spilling out of the gallbladder or biliary ducts. \
        - 3. Thermal injury is defined as an unintentional burn that leads to injury of non-target tissue. \
        - 4. Perforation is defined any tool tissue interaction that leads to perforation of the gallbladder or biliary ducts and the spillage of bile. \
        Use this JSON schema: {"error_type": int} with the type of error (1 for Bleeding, 2 for Bile Spillage, 3 for Thermal Injury, 4 for Perforation) and avoid line breaks. Only return this JSON.'
        PROMPTS[('heichole_error_recognition', 'GPT4o')] = PROMPTS[('heichole_error_recognition', 'GeminiPro1-5')]
        
        # Cholec80 Error Classification
        PROMPTS[('cholec80_error_recognition', 'Phi-3.5-Vision')] = PROMPTS[('heichole_error_recognition', 'Phi-3.5-Vision')]
        PROMPTS[('cholec80_error_recognition', 'InternVL2-8B')] = PROMPTS[('heichole_error_recognition', 'Phi-3.5-Vision')]
        PROMPTS[('cholec80_error_recognition', 'Qwen2-VL-7B-Instruct')] = PROMPTS[('heichole_error_recognition', 'Phi-3.5-Vision')]
        PROMPTS[('cholec80_error_recognition', 'GeminiPro1-5')] = PROMPTS[('heichole_error_recognition', 'GeminiPro1-5')]
        PROMPTS[('cholec80_error_recognition', 'GPT4o')] = PROMPTS[('heichole_error_recognition', 'GeminiPro1-5')]

        # Heichole Error Detection
        PROMPTS[('heichole_error_detection', 'GeminiPro1-5')] = 'You are a helpful medical video assistant. \
        Task: Find where <ERROR_TYPE> occurs in the provided frames from a cholecystectomy video. \
        The errors include: \
        - 1. Bleeding is defined as blood flowing or moving from the source of injury that is clearly visible on the screen.\
        - 2. Bile spillage is defined as containing the first tool tissue interaction that leads to perforation of the gallbladder or biliary ducts and the spillage of bile. \
        Instructions: Assess these frames and estimate the timestamps (minutes and seconds) of when the error begins and ends. \
        Assume that the video is recoded at 10 fps and the error can be any duration of time between 0 and 3 minutes. \
        Use this JSON schema: {"start_time": "00:00", "end_time": "00:00"} and avoid line breaks. \
        Make sure to give precise timestamps. Only return this JSON.'
        PROMPTS[('heichole_error_detection', 'Phi-3.5-Vision')] = '''You are an assistant skilled in analyzing surgical video data. Your task is to locate the time span during which a specific error (<ERROR_TYPE>) occurs in a series of frames extracted from a cholecystectomy video. The error definitions are as follows:
        1. Bleeding: Visible blood flowing or moving from the injury source on screen.
        2. Bile spillage: The initial tool-tissue interaction that causes perforation of the gallbladder or biliary ducts, resulting in bile leakage.
        Task: You are provided a series of <NUM_SAMP> frames sampled from a cholecystectomy video. Find the start and end frame numbers of the error. 
        Return your result using this JSON format (without any line breaks):
        {"start": int, "end": int}. Only return this JSON.'''
        PROMPTS[('heichole_error_detection', 'InternVL2-8B')] = PROMPTS[('heichole_error_detection', 'Phi-3.5-Vision')]
        PROMPTS[('heichole_error_detection', 'GPT4o')] = PROMPTS[('heichole_error_detection', 'Phi-3.5-Vision')]
        PROMPTS[('heichole_error_detection', 'Qwen2-VL-7B-Instruct')] = '''
        This video is 3 minutes long.
        Each frame is associated with a specific timestamp using the format 'mm:ss'.
        Here are the frames and their timestamps:
        Frame 0: 00:00
        Frame 1: 00:51
        Frame 2: 01:42
        ...
        Frame 35: 03:00 (max timestamp)
        Given the query: '<ERROR_TYPE>', when does the described content occur in the video?
        Use the 'mm:ss' format for your answer. Return in JSON format: {"start": mm:ss, "end": mm:ss}. 
        Only return this JSON.'''

        # Cholec80 Error Detection
        PROMPTS[('cholec80_error_detection', 'GeminiPro1-5')] = PROMPTS[('heichole_error_detection', 'GeminiPro1-5')]
        PROMPTS[('cholec80_error_detection', 'Phi-3.5-Vision')] = PROMPTS[('heichole_error_detection', 'Phi-3.5-Vision')]
        PROMPTS[('cholec80_error_detection', 'InternVL2-8B')] = PROMPTS[('heichole_error_detection', 'InternVL2-8B')]
        PROMPTS[('cholec80_error_detection', 'GPT4o')] = PROMPTS[('heichole_error_detection', 'GPT4o')]
        PROMPTS[('cholec80_error_detection', 'Qwen2-VL-7B-Instruct')] = PROMPTS[('heichole_error_detection', 'Qwen2-VL-7B-Instruct')]

        # Multibypass Phase
        PROMPTS[('multibypass140_phase_recognition', 'GeminiPro1-5')] = 'You are shown an image captured during a laparoscopic gastric bypass surgery. Determine the surgical phase of the image. The possible phases are \
        0: Preparation, 1: Gastric pouch creation, 2: Omentum division, 3: Gastrojejunal anastomosis, 4: Anastomosis test, 5: Jejunal separation, 6:Petersen space closure, 7: Jejunojejunal anastomosis, \
        8: Mesenteric defect closure, 9: Cleaning & Coagulation, 10: Disassembling, 11: Other intervention. There are no other options. Use this JSON schema: {"phase": int} and avoid line breaks.'
        
        multi140_phases = ['preparation', 'gastric pouch creation', 'omentum division', 'gastrojejunal anastomosis', 'anastomosis test', 'jejunal separation', 'petersen space closure', 'jejunojejunal anastomosis', \
                'mesenteric defect closure', 'cleaning & coagulation', 'disassembling', 'other intervention']
        candidate_captions_positive = ["A surgical scene during %s." % cls for cls in multi140_phases]
        PROMPTS[('multibypass140_phase_recognition', 'CLIP')] = candidate_captions_positive

        PROMPTS[('multibypass140_phase_recognition', 'SurgVLP')] = [  # prompts generated with gpt based on cholec80 surgvlp author given prompt
        'In preparation phase I insert trocars to patient abdomen cavity and prepare the surgical instruments',
        'In gastric pouch creation phase I use stapler to create a small gastric pouch from the stomach',
        'In omentum division phase I divide the omentum to prepare the space for the bypass',
        'In gastrojejunal anastomosis phase I connect the gastric pouch to the jejunum using a stapler or sutures',
        'In anastomosis test phase I test the anastomosis for leaks by injecting saline and observing for any leakage',
        'In jejunal separation phase I use stapler to separate the jejunum at the appropriate length for the bypass',
        'In petersen space closure phase I close the Petersen space to prevent internal hernia formation',
        'In jejunojejunal anastomosis phase I connect the proximal and distal parts of the jejunum to ensure intestinal continuity',
        'In mesenteric defect closure phase I close the mesenteric defect to prevent internal hernias',
        'In cleaning & coagulation phase I use suction and irrigation to clear the surgical field and coagulate bleeding vessels',
        'In disassembling phase I remove the surgical instruments and prepare to close the abdomen',
        'In other intervention phase I address any additional surgical requirements or complications as needed'
        ]

        PROMPTS[('multibypass140_phase_recognition', 'paligemma-3b-mix-448')] = "Answer en What is the surgical phase depicted in this image? Choose one answer from this list: Preparation, Gastric Pouch Creation, Omentum Division, Gastrojejunal Anastomosis, Anastomosis Test, Jejunal Separation, Petersen Space Closure, Jejunojejunal Anastomosis, Mesenteric Defect Closure, Cleaning and Coagulation, Disassembling, Other Intervention."

        # CholecT45 Action Triplets

        PROMPTS[('cholect45_triplet_recognition', 'GeminiPro1-5')] = 'Find all instruments in these images of laparoscopic cholecystectomies. For each instrument, provide the action \
                                it is performing and the tissue it is performing the action on. The following instruments are possible: \
                                {grasper, bipolar, hook, scissors, clipper, irrigator, null}. Choose one of the following actions: {grasp, retract, dissect, coagulate, \
                                clip, cut, aspirate, irrigate, pack, null}. Choose one of the following tissues: {gallbladder, cystic plate, cystic duct, \
                                cysic artery, cystic pedicle, blood vessel, fluid, abdominal wall cavity, liver, adhesion, omentum, peritoneum, gut, specimen bag, null}. \
                                Return a  dict using this JSON schema: {"instrument": [tool1,...], "verb": [activity1,...], "target": [tissue1,...]}, \
                                and avoid line breaks. If no instrument is present, return {"instrument": ["null"], "verb": ["null"], "target": ["null"]}. If an \
                                instrument is present but no activity or tissue is visible, return {"instrument": ["tool1", ...], "verb": ["null"], "target": ["null"]}.'


        # Note: PaliGemma prompt is dynamically generated in eval loop; below is a placeholder:
        PROMPTS[('cholect45_triplet_recognition', 'paligemma-3b-mix-448')] = ["Answer en Is there a grasper in this image?", "Is there a bipolar in this image?", "Is there a hook in this image?", "Are there scissors in this image?", "Is there a clipper in this image?", "Is there a irrigator in this image?", "Is there a specimen bag in this image?"]

        ## for SurgVLP
        if model == 'SurgVLP' and task == 'cholect45_triplet_recognition':
                triplet_file = path + 'dict/triplet.txt'
                triplet_prompts = []
                with open(triplet_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                                line = line.replace('cut', 'cutt').replace('clip', 'clipp')
                                if "null_instrument" in line:
                                        triplet_prompts.append("I don't use tools.")
                                elif "null_verb" in line:
                                        line = line.strip().split(':')[1].split(',')
                                        if "scissors" in line:
                                                triplet_prompts.append("I use scissors that do not do anything.")
                                        elif "irrigator" in line:
                                                triplet_prompts.append("I use an irrigator that does not do anything.")
                                        else:
                                                triplet_prompts.append("I use a %s that is not doing anything." % line[0])
                                elif "scissors" in line:
                                        line = line.strip().split(':')[1].split(',')
                                        triplet_prompts.append("I use scissors to %s the %s." % (line[1], line[2]))
                                elif "irrigator" in line:
                                        line = line.strip().split(':')[1].split(',')
                                        triplet_prompts.append("I use an irrigator to %s the %s." % (line[1], line[2]))
                                else:
                                        line = line.strip().split(':')[1].split(',')
                                        triplet_prompts.append("I use a %s to %s the %s." % (line[0], line[1], line[2]))
                PROMPTS[('cholect45_triplet_recognition', 'SurgVLP')] = triplet_prompts

        ## for CLIP
        if model == 'CLIP' and task == 'cholect45_triplet_recognition':
                triplet_file = path + 'dict/triplet.txt'
                triplet_prompts = []
                with open(triplet_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                                line = line.replace('cut', 'cutt').replace('clip', 'clipp')
                                if "null_instrument" in line:
                                        triplet_prompts.append("A surgical scene without any tools.")
                                elif "null_verb" in line:
                                        line = line.strip().split(':')[1].split(',')
                                        if "scissors" in line:
                                                triplet_prompts.append("Scissors that are not touching any tissue.")
                                        elif "irrigator" in line:
                                                triplet_prompts.append("An irrigator that is not touching any tissue.")
                                        else:
                                                triplet_prompts.append("A %s that is not touching any tissue." % line[0])
                                elif "scissors" in line:
                                        line = line.strip().split(':')[1].split(',')
                                        triplet_prompts.append("Scissors are %sing the %s." % (line[1].strip('e'), line[2]))
                                elif "irrigator" in line:
                                        line = line.strip().split(':')[1].split(',')
                                        triplet_prompts.append("An irrigator is %sing the %s." % (line[1].strip('e'), line[2]))
                                else:
                                        line = line.strip().split(':')[1].split(',')
                                        triplet_prompts.append("A %s is %sing the %s." % (line[0], line[1].strip('e'), line[2]))
                PROMPTS[('cholect45_triplet_recognition', 'CLIP')] = triplet_prompts

        # copy remaining prompts to other models
        for tuple in list(PROMPTS):
                try:
                        task_, model_ = tuple
                        if model_ == 'GeminiPro1-5': # copy Gemini prompt to other models
                                # but dont overwrite if exists (this means the model is handled individually)
                                if not (task_, 'GPT4o') in PROMPTS:
                                        PROMPTS[(task_, 'GPT4o')] = PROMPTS[(task_, model_)]
                                if not (task_, 'Qwen2-VL-7B-Instruct') in PROMPTS:
                                        PROMPTS[(task_, 'Qwen2-VL-7B-Instruct')] = PROMPTS[(task_, model_)]
                                if not (task_, 'llava_next_vicuna_7b') in PROMPTS:
                                        PROMPTS[(task_, 'llava_next_vicuna_7b')] = PROMPTS[(task_, model_)]
                        elif model_ == 'CLIP':
                               PROMPTS[(task_, 'OpenCLIP')] = PROMPTS[(task_, model_)] 
                except:
                        pass
                                
        
        # JIGSAWS skill assessment — one dimension per task, single score output (matches paper protocol)
        # respect_for_tissue (default)
        PROMPTS[('jigsaws_skill_assessment', 'Qwen2-VL-7B-Instruct')] = 'You are a helpful medical video assistant. \
        Task: assess the respect for tissue on a scale from 1 to 5. Use the following criteria to output the score: \
        1-Frequently used unnecessary force on tissue or caused damage by inappropriate use of instruments; \
        3-Careful tissue handling but occasionally caused inadvertent damage; \
        5-Consistently handled tissues appropriately with minimal damage; \
        Instructions: assess the video carefully, and respond with the respect for tissue score. Only output the score.'
        PROMPTS[('jigsaws_skill_assessment', 'Phi-3.5-Vision')] = 'You are a helpful medical video assistant. You will be provided with separate frames uniformaly sampled from a video. \
        Task: assess the respect for tissue on a scale from 1 to 5. Use the following criteria to output the score: \
        1-Frequently used unnecessary force on tissue or caused damage by inappropriate use of instruments; \
        3-Careful tissue handling but occasionally caused inadvertent damage; \
        5-Consistently handled tissues appropriately with minimal damage; \
        Instructions: assess the video carefully, and respond with the respect for tissue score. Only output the score.'
        # suture_needle_handling
        PROMPTS[('jigsaws_skill_assessment_suture_needle_handling', 'Qwen2-VL-7B-Instruct')] = 'You are a helpful medical video assistant. \
        Task: assess the suture / needle handling on a scale from 1 to 5. Use the following criteria to output the score: \
        1-Repeatedly awkward and unsure with needle, entanglement and poor knot tying; \
        3-Competent use of needle and suture, majority of knots placed correctly with appropriate tension; \
        5-Fluid moves with needle and suture, excellent knot tying and instrument control; \
        Instructions: assess the video carefully, and respond with the suture / needle handling score. Only output the score.'
        PROMPTS[('jigsaws_skill_assessment_suture_needle_handling', 'Phi-3.5-Vision')] = 'You are a helpful medical video assistant. You will be provided with separate frames uniformaly sampled from a video. \
        Task: assess the suture / needle handling on a scale from 1 to 5. Use the following criteria to output the score: \
        1-Repeatedly awkward and unsure with needle, entanglement and poor knot tying; \
        3-Competent use of needle and suture, majority of knots placed correctly with appropriate tension; \
        5-Fluid moves with needle and suture, excellent knot tying and instrument control; \
        Instructions: assess the video carefully, and respond with the suture / needle handling score. Only output the score.'
        # time_and_motion
        PROMPTS[('jigsaws_skill_assessment_time_and_motion', 'Qwen2-VL-7B-Instruct')] = 'You are a helpful medical video assistant. \
        Task: assess the time and motion on a scale from 1 to 5. Use the following criteria to output the score: \
        1-Many unnecessary moves, inefficient; \
        3-Efficient time and motion but some unnecessary moves; \
        5-Clear economy of movement and maximum efficiency; \
        Instructions: assess the video carefully, and respond with the time and motion score. Only output the score.'
        PROMPTS[('jigsaws_skill_assessment_time_and_motion', 'Phi-3.5-Vision')] = 'You are a helpful medical video assistant. You will be provided with separate frames uniformaly sampled from a video. \
        Task: assess the time and motion on a scale from 1 to 5. Use the following criteria to output the score: \
        1-Many unnecessary moves, inefficient; \
        3-Efficient time and motion but some unnecessary moves; \
        5-Clear economy of movement and maximum efficiency; \
        Instructions: assess the video carefully, and respond with the time and motion score. Only output the score.'
        # flow_of_operation
        PROMPTS[('jigsaws_skill_assessment_flow_of_operation', 'Qwen2-VL-7B-Instruct')] = 'You are a helpful medical video assistant. \
        Task: assess the flow of operation on a scale from 1 to 5. Use the following criteria to output the score: \
        1-Frequently stopped operating or needed to discuss next move; \
        3-Demonstrated some forward planning with reasonable progression of procedure; \
        5-Obviously planned course of operation with effortless flow from one move to the next; \
        Instructions: assess the video carefully, and respond with the flow of operation score. Only output the score.'
        PROMPTS[('jigsaws_skill_assessment_flow_of_operation', 'Phi-3.5-Vision')] = 'You are a helpful medical video assistant. You will be provided with separate frames uniformaly sampled from a video. \
        Task: assess the flow of operation on a scale from 1 to 5. Use the following criteria to output the score: \
        1-Frequently stopped operating or needed to discuss next move; \
        3-Demonstrated some forward planning with reasonable progression of procedure; \
        5-Obviously planned course of operation with effortless flow from one move to the next; \
        Instructions: assess the video carefully, and respond with the flow of operation score. Only output the score.'
        # overall_performance
        PROMPTS[('jigsaws_skill_assessment_overall_performance', 'Qwen2-VL-7B-Instruct')] = 'You are a helpful medical video assistant. \
        Task: assess the overall performance on a scale from 1 to 5. Use the following criteria to output the score: \
        1-Very poor, unable to perform the procedure; \
        3-Competent, could be allowed to perform procedure under supervision; \
        5-Clearly superior, could perform the procedure independently; \
        Instructions: assess the video carefully, and respond with the overall performance score. Only output the score.'
        PROMPTS[('jigsaws_skill_assessment_overall_performance', 'Phi-3.5-Vision')] = 'You are a helpful medical video assistant. You will be provided with separate frames uniformaly sampled from a video. \
        Task: assess the overall performance on a scale from 1 to 5. Use the following criteria to output the score: \
        1-Very poor, unable to perform the procedure; \
        3-Competent, could be allowed to perform procedure under supervision; \
        5-Clearly superior, could perform the procedure independently; \
        Instructions: assess the video carefully, and respond with the overall performance score. Only output the score.'

        # JIGSAWS gesture classification
        PROMPTS[('jigsaws_gesture_classification', 'Qwen2-VL-7B-Instruct')] = 'You are a helpful medical video assistant. \
        Task: classify the gesture of the surgical activity video segment. Below are the defined gestures: \
        G1 Reaching for needle with right hand; \
        G2 Positioning needle; \
        G3 Pushing needle through tissue; \
        G4 Transferring needle from left to right; \
        G5 Moving to center with needle in grip; \
        G6 Pulling suture with left hand; \
        G7 Pulling suture with right hand; \
        G8 Orienting needle; \
        G9 Using right hand to help tighten suture; \
        G10 Loosening more suture; \
        G11 Dropping suture at end and moving to end points; \
        G12 Reaching for needle with left hand; \
        G13 Making C loop around right hand; \
        G14 Reaching for suture with right hand; \
        G15 Pulling suture with both hands.; \
        Instructions: Assess the video segment carefully and classify the gesture. The segment only contains one gesture. Only output the gesture, eg: G1'
        PROMPTS[('jigsaws_gesture_classification', 'Phi-3.5-Vision')] = 'You are a helpful medical video assistant. You will be provided with separate frames uniformaly sampled from a video segment. \
        Task: classify the gesture of the surgical activity video segment. Below are the defined gestures: \
        G1 Reaching for needle with right hand; \
        G2 Positioning needle; \
        G3 Pushing needle through tissue; \
        G4 Transferring needle from left to right; \
        G5 Moving to center with needle in grip; \
        G6 Pulling suture with left hand; \
        G7 Pulling suture with right hand; \
        G8 Orienting needle; \
        G9 Using right hand to help tighten suture; \
        G10 Loosening more suture; \
        G11 Dropping suture at end and moving to end points; \
        G12 Reaching for needle with left hand; \
        G13 Making C loop around right hand; \
        G14 Reaching for suture with right hand; \
        G15 Pulling suture with both hands.; \
        Instructions: Assess the images carefully and classify the gesture. The segment only contains one gesture. Only output the gesture, eg: G1.'

        # HeiChole skill assessment — all dimensions in one JSON output
        PROMPTS[('heichole_skill_assessment', 'Qwen2-VL-7B-Instruct')] = 'You are a helpful medical video assistant. \
        Task: assess the surgical skill of a laparoscopic cholecystectomy on a scale from 1 to 5 for each of the following dimensions. Use the criteria below: \
        - depth_perception: 1=Constantly overshoots target, wide swings, slow to correct; 3=Some overshooting but quick to correct; 5=Accurately directs instruments to target. \
        - bimanual_dexterity: 1=Uses only one hand, poor coordination; 3=Uses both hands but not optimized; 5=Expertly uses both hands for optimal exposure. \
        - efficiency: 1=Uncertain, inefficient, many tentative movements; 3=Slow but planned and organized; 5=Confident, efficient, and safe conduct. \
        - tissue_handling: 1=Rough movements, tears tissue, poor grasper control; 3=Handles tissues reasonably, minor trauma; 5=Handles tissues well, negligible injury. \
        - difficulty: 1=Easy exploration and dissection; 3=Moderate difficulty (mild inflammation, scarring); 5=Extremely difficult (severe inflammation, scarring). \
        Instructions: assess the video carefully. Use this JSON schema: {"depth_perception": int, "bimanual_dexterity": int, "efficiency": int, "tissue_handling": int, "difficulty": int} and avoid line breaks.'
        PROMPTS[('heichole_skill_assessment', 'Phi-3.5-Vision')] = PROMPTS[('heichole_skill_assessment', 'Qwen2-VL-7B-Instruct')]


        # Autolaporo maneuver classification
        PROMPTS[('autolaporo_maneuver_classification', 'Qwen2-VL-7B-Instruct')] = 'You are a helpful medical video assistant. \
        Task: predict the laparoscope motion that will occur immediately after the video. The seven types of defined motion are: \
        Static, Up, Down, Left, Right, Zoom-in, Zoom-out. \
        The future movement will be made to ensure proper field-of-view for the surgeon. If no movement is needed, then output Static. \
        Instructions: assess the video carefully, and respond with the future laparoscope movement. Only output one of the given motions, and do not explain why.'
        PROMPTS[('autolaporo_maneuver_classification', 'Phi-3.5-Vision')] = 'You are a helpful medical video assistant. You will be provided with separate frames uniformaly sampled from a video. \
        Task: predict the laparoscope motion that will occur immediately after the video. The seven types of defined motion are: \
        Static, Up, Down, Left, Right, Zoom-in, Zoom-out. \
        The future movement will be made to ensure proper field-of-view for the surgeon. If no movement is needed, then output Static. \
        Instructions: assess the video carefully, and respond with the future laparoscope movement. Only output one of the given motions, and do not explain why.'

        # ── CholecT50 Phase Planning ──────────────────────────────────

        _phase_list = (
            'The procedure follows these phases in order:\n'
            '  [1] Preparation\n'
            '  [2] Calot Triangle Dissection\n'
            '  [3] Clipping & Cutting\n'
            '  [4] Gallbladder Dissection\n'
            '  [5] Gallbladder Packaging\n'
            '  [6] Cleaning & Coagulation\n'
            '  [7] Gallbladder Retraction'
        )

        _phase_landmarks = (
            '  [1] Preparation:\n'
            '      Early: Trocar insertion, initial camera positioning.\n'
            '      Middle: Grasper positioning on gallbladder fundus, initial retraction.\n'
            '      Late: Good retraction established, gallbladder lifted, field exposed.\n'
            '  [2] Calot Triangle Dissection:\n'
            '      Early: Initial dissection near gallbladder neck, hook approaching Calot\'s triangle.\n'
            '      Middle: Clearing fat/tissue from Calot\'s triangle, cystic duct partially exposed.\n'
            '      Late: Cystic duct and artery fully exposed, Critical View of Safety achieved.\n'
            '  [3] Clipping & Cutting:\n'
            '      Early: Clipper introduced, approaching cystic duct.\n'
            '      Middle: Cystic duct clipped, moving to clip cystic artery.\n'
            '      Late: Both structures clipped and divided.\n'
            '  [4] Gallbladder Dissection:\n'
            '      Early: Hook dissecting gallbladder from liver bed at the neck.\n'
            '      Middle: Gallbladder partially detached, progressing along liver bed.\n'
            '      Late: Nearly fully detached, thin attachments remaining.\n'
            '  [5] Gallbladder Packaging:\n'
            '      Early: Specimen bag being introduced.\n'
            '      Middle: Gallbladder being maneuvered toward bag opening.\n'
            '      Late: Gallbladder placed in bag, being closed.\n'
            '  [6] Cleaning & Coagulation:\n'
            '      Early: Irrigator washing liver bed, inspecting for bleeding.\n'
            '      Middle: Active coagulation with bipolar, irrigation/aspiration.\n'
            '      Late: Liver bed dry, minimal bleeding.\n'
            '  [7] Gallbladder Retraction:\n'
            '      Early: Specimen bag being pulled toward trocar site.\n'
            '      Middle: Extraction through port.\n'
            '      Late: Gallbladder fully extracted, instruments removed.'
        )

        _transition_rule = (
            'For next_phase:\n'
            '  - If the phase appears to be ongoing (not near completion), next_phase = current_phase.\n'
            '  - If the phase appears to be finishing or transitioning, next_phase = the following phase.'
        )

        # No CoT — direct prediction
        _phase_planning_prompt = (
            'You are monitoring a laparoscopic cholecystectomy.\n'
            + _phase_list + '\n\n'
            'You are shown consecutive frames from the procedure.\n'
            'Identify the current phase, and predict the next phase.\n\n'
            + _transition_rule + '\n\n'
            'Respond in JSON: {{"current_phase": "...", "next_phase": "..."}}'
        )

        # With CoT — landmarks + structured reasoning
        _phase_planning_cot_prompt = (
            'You are monitoring a laparoscopic cholecystectomy.\n'
            'The procedure follows these phases in order, with typical visual landmarks:\n\n'
            + _phase_landmarks + '\n\n'
            'You are shown consecutive frames from the procedure.\n'
            'First identify which phase and stage you see, then predict what comes next.\n\n'
            + _transition_rule + '\n\n'
            'Respond in JSON with ALL fields:\n'
            '{{\n'
            '  "instruments": ["list visible instruments"],\n'
            '  "actions": ["describe what each instrument is doing"],\n'
            '  "current_phase": "...",\n'
            '  "stage": "early or middle or late",\n'
            '  "next_phase": "..."\n'
            '}}'
        )

        # Progress prediction prompt
        _phase_progress_prompt = (
            'You are monitoring a laparoscopic cholecystectomy.\n'
            + _phase_list + '\n\n'
            'You are shown consecutive frames from the procedure.\n'
            'Identify the current surgical phase, then estimate how far along that phase is.\n\n'
            'Respond in JSON: {{"current_phase": "...", "progress": "early or middle or late"}}\n\n'
            'early = phase just started (0-33%%), middle = phase underway (33-67%%), late = phase nearly done (67-100%%)'
        )

        _all_models = ['GeminiPro1-5', 'Qwen2-VL-7B-Instruct', 'InternVL3-8B', 'MedGemma-4B',
                        'Qwen3-VL-8B-Instruct', 'Qwen3-VL-8B-Thinking', 'llava_next_vicuna_7b']

        # Register for all datasets x all models
        _planning_tasks = ['cholect50_phase_planning', 'cholec80_phase_planning', 'heichole_phase_planning']
        _progress_tasks = ['cholect50_phase_progress_prediction', 'cholec80_phase_progress_prediction', 'heichole_phase_progress_prediction']

        for _model in _all_models:
            for _task in _planning_tasks:
                PROMPTS[(_task, _model)] = _phase_planning_prompt
            for _task in _progress_tasks:
                PROMPTS[(_task, _model)] = _phase_progress_prompt

        # ── VTRB Suturing Phase Predict Easy ─────────────────────────────────
        # Template: {choices} is filled per-sample in inference
        _vtrb_phase_easy_prompt = (
            'You are observing a robotic suturing procedure.\n'
            'The image shows a step from the suturing task.\n\n'
            'Which of the following phases is being performed?\n'
            '{choices}\n\n'
            'Respond with ONLY the letter (A, B, C, D, or E).'
        )

        _vtrb_phase_planning_prompt = (
            'You are observing a robotic suturing procedure.\n'
            'You are shown consecutive frames from the procedure.\n\n'
            'The possible phases are:\n'
            '{choices}\n\n'
            'Identify the current phase, then predict the next phase.\n'
            'If the current phase is still ongoing, next_phase should be the same as current_phase.\n'
            'If the phase is finishing, next_phase should be the following phase.\n\n'
            'Respond in JSON: {{"current_phase": "A or B or C or D or E", "next_phase": "A or B or C or D or E"}}'
        )

        for _model in _all_models:
            PROMPTS[('vtrb_suturing_phase_predict_easy', _model)] = _vtrb_phase_easy_prompt
            PROMPTS[('vtrb_suturing_phase_planning', _model)] = _vtrb_phase_planning_prompt

        # ── Phase + Triplet Planning ──────────────────────────────────────────
        _triplet_planning_prompt = (
            'You are a surgical assistant monitoring a laparoscopic cholecystectomy.\n'
            + _phase_list + '\n\n'
            'VOCABULARY:\n'
            '  Instruments: grasper, bipolar, hook, scissors, clipper, irrigator\n'
            '  Verbs: grasp, retract, dissect, coagulate, clip, cut, aspirate, irrigate, pack\n'
            '  Targets: gallbladder, cystic_plate, cystic_duct, cystic_artery, cystic_pedicle,\n'
            '           blood_vessel, fluid, abdominal_wall_cavity, liver, adhesion, omentum,\n'
            '           peritoneum, gut, specimen_bag\n\n'
            'You are shown consecutive frames from the procedure.\n'
            'Identify what is happening and predict what comes next.\n\n'
            + _transition_rule + '\n\n'
            'Respond in JSON:\n'
            '{{\n'
            '  "phase": "...",\n'
            '  "current_triplet": [{{"instrument": "...", "verb": "...", "target": "..."}}],\n'
            '  "next_phase": "...",\n'
            '  "next_triplet": [{{"instrument": "...", "verb": "...", "target": "..."}}]\n'
            '}}'
        )

        for _model in ['GeminiPro1-5', 'Qwen2-VL-7B-Instruct', 'InternVL3-8B', 'MedGemma-4B',
                        'Qwen3-VL-8B-Instruct', 'Qwen3-VL-8B-Thinking', 'llava_next_vicuna_7b']:
            PROMPTS[('cholect50_phase_triplet_planning', _model)] = _triplet_planning_prompt

        # ── VTRB-Suturing Recognition (few-shot bbox) ────────────────────
        _vtrb_recognition_prompt = (
            'Now detect all objects in this NEW image. '
            'Use the same classes and bbox format as the examples above.\n\n'
            'Respond in JSON:\n'
            '{{"objects": [{{"class": "...", "bbox": [x1, y1, x2, y2], "description": "..."}}]}}'
        )
        for _model in ['GeminiPro1-5', 'Qwen2-VL-7B-Instruct', 'InternVL3-8B', 'MedGemma-4B',
                        'Qwen3-VL-8B-Instruct', 'Qwen3-VL-8B-Thinking', 'llava_next_vicuna_7b',
                        'paligemma-3b-mix-448']:
            PROMPTS[('vtrb_suturing_recognition', _model)] = _vtrb_recognition_prompt

        # Copy Phi-3.5-Vision prompt to other models for video tasks
        for tuple in list(PROMPTS):
                try:
                        task_, model_ = tuple
                        if model_ == 'Phi-3.5-Vision' and ('jigsaws' in task_ or task_ == 'autolaporo_maneuver_classification' or 'heichole_skill_assessment' in task_):
                                PROMPTS[(task_, 'InternVL2-8B')] = PROMPTS[(task_, model_)]
                                PROMPTS[(task_, 'Llama-3-VILA1.5-8b')] = PROMPTS[(task_, model_)]
                                PROMPTS[(task_, 'GeminiPro1-5')] = PROMPTS[(task_, model_)]
                                PROMPTS[(task_, 'GPT4o')] = PROMPTS[(task_, model_)]
                except:
                        pass

        # Copy Qwen2-VL skill/video prompts to video-capable models
        video_aliases = ['InternVL3-8B', 'MedGemma-4B', 'Qwen3-VL-8B-Instruct', 'Qwen3-VL-8B-Thinking']
        qwen2_video_tasks = [k for k in PROMPTS if k[1] == 'Qwen2-VL-7B-Instruct' and ('skill' in k[0] or 'jigsaws' in k[0])]
        for task_name, _ in qwen2_video_tasks:
                for alias in video_aliases:
                        if (task_name, alias) not in PROMPTS:
                                PROMPTS[(task_name, alias)] = PROMPTS[(task_name, 'Qwen2-VL-7B-Instruct')]

        # Alias GeminiPro1-5 prompts for generative models that use the same JSON format
        generative_aliases = ['Qwen2-VL-7B-Instruct', 'InternVL3-8B', 'MedGemma-4B', 'Qwen3-VL-8B-Instruct', 'Qwen3-VL-8B-Thinking']
        gemini_tasks = [k for k in PROMPTS if k[1] == 'GeminiPro1-5']
        for task_name, _ in gemini_tasks:
                for alias in generative_aliases:
                        if (task_name, alias) not in PROMPTS:
                                PROMPTS[(task_name, alias)] = PROMPTS[(task_name, 'GeminiPro1-5')]

        # Alias llava action prompt for other generative models too
        llava_tasks = [k for k in PROMPTS if k[1] == 'llava_next_vicuna_7b']
        for task_name, _ in llava_tasks:
                for alias in generative_aliases:
                        if (task_name, alias) not in PROMPTS:
                                PROMPTS[(task_name, alias)] = PROMPTS[(task_name, 'llava_next_vicuna_7b')]

        return PROMPTS[(task, model)]