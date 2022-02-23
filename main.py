from google.cloud import speech_v1
import io
import subprocess
from pydub import AudioSegment
import imutils
import dlib
from imutils import face_utils
import cv2
from moviepy.editor import *
from profanity_check import predict


def speech_recognize(local_file_path):

    client = speech_v1.SpeechClient()

    enable_word_time_offsets = True

    language_code = "en-US"

    sample_rate_hertz = 16000

    config = {
        "enable_word_time_offsets": enable_word_time_offsets,  # 开启时间戳
        "language_code": language_code,
        "sample_rate_hertz": sample_rate_hertz,
    }
    with io.open(local_file_path, "rb") as f:
        content = f.read()
    audio = {"content": content}

    operation = client.long_running_recognize(config, audio)

    print(u"Waiting for operation to complete...")
    response = operation.result()
    result = response.results[0]

    alternative = result.alternatives[0]
    print(u"Transcript: {}".format(alternative.transcript))

    timelist = []

    checker = predict([alternative.transcript])  # 对文本进行检测

    vec = [0]*len(alternative.words)

    if checker[0] == 1:  # 如果是亵渎性语句 检测短语
        with open('s_words3.txt', 'r') as f:
            rd = f.read().splitlines()
        text = str(alternative.transcript)
        for x in rd:
            if x in text:
                pos = text.find(x)
                fr_text = text[:pos]
                num = fr_text.count(' ')
                cnt = x.count(' ')
                for i in range(num, num+cnt+1):
                    vec[i] = 1

    with open('s_words2.txt', 'r') as f:  # 检测单词
        rd = f.read().splitlines()
        for i in range(len(alternative.words)):
            if alternative.words[i].word in rd:
                vec[i] = 1

    cont = False
    st_t = 0  # 开始时间
    ed_t = 0  # 结束时间
    for i in range(len(alternative.words)):
        if vec[i] == 0:
            if cont:
                timelist.append((st_t, ed_t))
                cont = False
        else:
            if not cont:  # 连续 都需要转换为ms
                st_t = 1000*(alternative.words[i].start_time.seconds+alternative.words[i].start_time.nanos/1.0e9)
                ed_t = 1000*(alternative.words[i].end_time.seconds+alternative.words[i].end_time.nanos/1.0e9)
                cont = True
            else:
                ed_t = 1000*(alternative.words[i].end_time.seconds+alternative.words[i].end_time.nanos/1.0e9)
    if cont:
        timelist.append((st_t, ed_t))

    return timelist


def detect(detector, predictor, img, face_cascade, mstart, mend):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, 1.1, 3, 0, (30, 30))
    # for x, y, w, h in faces:
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 1)
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = imutils.face_utils.shape_to_np(shape)

        mouth = shape[mstart:mend]

        mouth_hull = cv2.convexHull(mouth)  # 取检测出轮廓的外接矩形
        max_x = max(mouth_hull[:, 0, 0])
        max_y = max(mouth_hull[:, 0, 1])
        min_x = min(mouth_hull[:, 0, 0])
        min_y = min(mouth_hull[:, 0, 1])
        width = max_x-min_x
        height = max_y-min_y
        roi = img[min_y-height//2:max_y+height//2, min_x-width//2:max_x+width//2]  # 略做扩大
        roi = cv2.GaussianBlur(roi, (95, 95), 0)  # 模糊
        img[min_y - height // 2:max_y + height // 2, min_x - width // 2:max_x + width // 2] = roi  # 替换

    return img


def video_processing(time, path):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    (mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('newvideo.avi', fourcc, fps, size, True)

    i = 0
    cont = False
    time_num = len(time)
    while True:
        ret, frame = cap.read()
        if not ret:  # 视频读完了
            break
        # 如果是需要屏蔽的部分
        if i < time_num and time[i][0] <= cap.get(cv2.CAP_PROP_POS_MSEC) < time[i][1]:
            frame = detect(detector, predictor, frame, face_cascade, mstart, mend)
            cont = True
        else:
            if cont:
                cont = False
                i += 1
        out.write(frame)

    cap.release()
    out.release()

    return 'newvideo.avi'


def get_audio(video_path):

    # moviepy.VideoFileClip: 读取视频到内存，返回一个VideoFileClip的对象
    video = VideoFileClip(video_path)
    # 获取视频的音频部分
    audio = video.audio
    pos = video_path.find('.')
    audio_path = video_path[:pos] + '.mp3'
    # 将视频中的音频部分提取出来，写入test.mp3
    audio.write_audiofile(audio_path)

    return audio_path


def audio_processing(time, path):

    length = len(time)
    # 读取需要处理的音频
    sound = AudioSegment.from_mp3(path)
    # 读取屏蔽音
    pb = AudioSegment.from_wav('bi.wav')
    ns = AudioSegment.empty()
    print(length)
    if length == 0:
        sound.export('word.mp3', format='mp3')  # 不需要屏蔽直接输出返回
        return 'word.mp3'
    if length == 1:  # 只有一段
        ns = sound[:time[0][0]] + pb[:time[0][1]-time[0][0]] + sound[time[0][1]:]
    else:  # 有多段 一段一段拼接
        for i in range(length):
            if i == 0:
                ns += sound[:time[0][0]] + pb[:time[0][1]-time[0][0]] + sound[time[0][1]:time[1][0]]
            elif i == length-1:
                ns += pb[:time[i][1]-time[i][0]] + sound[time[i][1]:]
            else:
                ns += pb[:time[i][1] - time[i][0]] + sound[time[i][1]:time[i+1][0]]
    ns.export('word.mp3', format='mp3')

    return 'word.mp3'


if __name__ == '__main__':

    path = '07.mp4'
    rm_list = []  # 需要删除的文件名列表

    audio_path = get_audio(path)
    rm_list.append(audio_path)

    time_list = speech_recognize(audio_path)
    print(time_list)

    video_path = video_processing(time_list, path)
    rm_list.append(video_path)

    audio_path2 = audio_processing(time_list, audio_path)
    rm_list.append(audio_path2)

    # 合并音频视频
    subprocess.call('ffmpeg -i ' + video_path
                    + ' -i ' + audio_path2 + ' -strict -2 -f mp4 -b:v 5390k '
                    + 'ress.mp4', shell=True)
    for path in rm_list:  # 删除中间文件
        subprocess.call('rm ' + path, shell=True)
