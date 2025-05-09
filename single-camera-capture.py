import cv2  # 导入 OpenCV 库，用于视频捕获和处理
import time  # 导入时间模块，用于时间戳和延迟
import logging  # 导入日志模块，用于记录运行状态
import argparse  # 导入命令行参数解析模块
from pathlib import Path  # 导入 Path，用于路径操作
from datetime import datetime  # 导入 datetime，用于获取当前时间戳
from queue import Queue  # 导入队列，用于线程间安全通信
from threading import Thread  # 导入线程模块，用于异步写入视频帧

# 四字符编码与文件扩展名映射字典
CODEC_EXT = {
    'mp4v': '.mp4',
    'XVID': '.avi',
    'MJPG': '.avi',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture video with timestamps and periodic snapshots into session folders.")  # 初始化参数解析器
    parser.add_argument("-o", "--output", default="output", help="Base output directory")  # 输出目录
    parser.add_argument("-f", "--fps", type=float, default=None,
                        help="Override FPS (if camera reports zero)")  # FPS 覆盖
    parser.add_argument("-c", "--camera", type=int, default=0, help="Camera index")  # 摄像头索引
    parser.add_argument("-s", "--snapshot", type=int, default=100,
                        help="Save a frame every N frames as image")  # 快照间隔
    parser.add_argument("--codec", default="XVID", help="4-character video codec")  # 编码格式
    parser.add_argument("--no-display", action="store_true", help="Disable real-time display")  # 禁用预览
    return parser.parse_args()  # 返回解析后的参数


def writer_thread_func(frame_queue: Queue, writer: cv2.VideoWriter):
    """
    后台写入线程函数，异步地将帧写入视频文件
    :param frame_queue: 存放待写入帧的队列
    :param writer: OpenCV 视频写入对象
    """
    while True:
        frame = frame_queue.get()  # 从队列获取帧
        if frame is None:
            break  # 收到退出信号，结束线程
        writer.write(frame)  # 写入帧到视频
        frame_queue.task_done()  # 标记任务完成
    writer.release()  # 释放写入对象资源


def main():
    args = parse_args()  # 获取命令行参数
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")  # 配置日志格式

    # 创建会话文件夹：使用当前时间戳作为子目录名
    base_output = Path(args.output)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = base_output / timestamp
    video_dir = session_dir
    snapshot_dir = session_dir / 'snapshots'
    photo_dir = session_dir / 'photos'  # 按下's'保存照片的文件夹
    for d in (video_dir, snapshot_dir, photo_dir):
        d.mkdir(parents=True, exist_ok=True)
    logging.info(f"Session folders created: {session_dir} (video, snapshots, photos)")

    cap = cv2.VideoCapture(args.camera)  # 打开摄像头
    if not cap.isOpened():
        logging.error(f"Cannot open camera {args.camera}")  # 打开失败
        return

    # 获取分辨率和帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = cap.get(cv2.CAP_PROP_FPS) or args.fps or 30.0  # 处理 FPS 为 0 的情况
    logging.info(f"Camera opened: {width}x{height} @ {cam_fps:.2f} FPS")

    # 准备视频写入器
    ext = CODEC_EXT.get(args.codec, '.avi')
    video_path = video_dir / f'recording{ext}'
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    out = cv2.VideoWriter(str(video_path), fourcc, cam_fps, (width, height))
    if not out.isOpened():
        logging.error(f"Cannot open video writer for {video_path}")
        cap.release()
        return

    # 创建写入线程及队列，提高写入效率
    frame_queue = Queue(maxsize=int(cam_fps) * 2)
    writer_thread = Thread(target=writer_thread_func, args=(frame_queue, out), daemon=True)
    writer_thread.start()

    frame_idx = 0  # 帧计数器
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Frame capture failed, stopping.")
                break

            # 在帧上叠加时间戳文字
            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            cv2.putText(frame, now_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # 将帧复制并放入队列以供写入
            try:
                frame_queue.put(frame.copy(), timeout=1)
            except Exception:
                logging.error("Frame queue is full, dropped frame.")

            # 定期保存快照
            if args.snapshot > 0 and frame_idx % args.snapshot == 0:
                snap_name = f"snap_{frame_idx}_{int(time.time())}.jpg"
                snap_path = snapshot_dir / snap_name
                cv2.imwrite(str(snap_path), frame)
                logging.info(f"Saved snapshot: {snap_path}")

            # 实时显示并处理按键
            if not args.no_display:
                cv2.imshow('Capture', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logging.info("Quit signal received.")
                    break
                elif key == ord('s'):
                    # 按下's'键，保存当前帧到photos文件夹
                    photo_name = f"photo_{frame_idx}_{int(time.time())}.jpg"
                    photo_path = photo_dir / photo_name
                    cv2.imwrite(str(photo_path), frame)
                    logging.info(f"Saved photo on 's' press: {photo_path}")

            frame_idx += 1
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")

    # 清理资源：停止捕获，通知写入线程退出，等待线程结束
    cap.release()
    frame_queue.put(None)
    writer_thread.join()
    cv2.destroyAllWindows()
    logging.info("All resources released. Exiting.")


if __name__ == '__main__':
    main()  # 执行主函数
