"""
增强版 YOLOv8 交通标志检测系统
新增功能: 视频检测、实时摄像头、批量处理、模型对比、性能分析
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import cv2
import numpy as np
import io
import tempfile
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 配置
CLASS_NAMES = [
    'Green Light', 'Red Light',
    'Speed Limit 10', 'Speed Limit 100', 'Speed Limit 110', 'Speed Limit 120',
    'Speed Limit 20', 'Speed Limit 30', 'Speed Limit 40', 'Speed Limit 50',
    'Speed Limit 60', 'Speed Limit 70', 'Speed Limit 80', 'Speed Limit 90',
    'Stop'
]

# 模型路径配置
MODELS = {
    '基线模型': 'runs/detect/yolov8n_baseline/weights/best.pt',
    '改进模型-数据增强': 'runs/detect/yolov8n_augment/weights/best.pt',
    # '改进模型-CBAM': 'runs/detect/yolov8n_cbam/weights/best.pt',
    '改进模型-CBAM': 'C:/app/cod_tool/program_tool/jet/pycharm_fil/25/1120\yolo0501/traffic_sign_data/runs/detect/yolov8n_cbam_method1/weights/best.pt',
}

@st.cache_resource
def load_model(model_path):
    """加载模型并缓存"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None


def detect_image(model, image, conf, iou, img_size):
    """图像检测"""
    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        imgsz=img_size,
        save_conf=True,
        verbose=False
    )
    return results[0]


def detect_video(model, video_path, conf, iou, img_size, progress_bar):
    """视频检测"""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 输出视频设置
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    detection_stats = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 检测
        results = model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            imgsz=img_size,
            verbose=False
        )

        # 绘制结果
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        # 统计
        num_detections = len(results[0].boxes)
        detection_stats.append({
            'frame': frame_count,
            'detections': num_detections
        })

        frame_count += 1
        progress_bar.progress(frame_count / total_frames)

    cap.release()
    out.release()

    return output_path, pd.DataFrame(detection_stats)


def batch_process_images(model, uploaded_files, conf, iou, img_size):
    """批量处理图片"""
    results_list = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"处理中: {uploaded_file.name} ({idx+1}/{len(uploaded_files)})")

        image = Image.open(uploaded_file)
        result = detect_image(model, image, conf, iou, img_size)

        # 统计结果
        num_detections = len(result.boxes)
        detected_classes = [CLASS_NAMES[int(box.cls[0])] for box in result.boxes]

        results_list.append({
            '文件名': uploaded_file.name,
            '检测数量': num_detections,
            '检测类别': ', '.join(set(detected_classes)) if detected_classes else '无'
        })

        progress_bar.progress((idx + 1) / len(uploaded_files))

    status_text.text("批量处理完成!")
    return pd.DataFrame(results_list)


def compare_models(image, models_dict, conf, iou, img_size):
    """对比多个模型"""
    comparison_results = []

    for model_name, model_path in models_dict.items():
        model = load_model(model_path)
        if model is None:
            continue

        start_time = time.time()
        result = detect_image(model, image, conf, iou, img_size)
        inference_time = time.time() - start_time

        comparison_results.append({
            '模型': model_name,
            '检测数量': len(result.boxes),
            '推理时间(s)': f"{inference_time:.3f}",
            'FPS': f"{1/inference_time:.1f}"
        })

        # 显示检测结果
        im_bgr = result.plot()
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        st.image(im_rgb, caption=model_name, use_container_width=True)

    return pd.DataFrame(comparison_results)


def plot_class_distribution(result):
    """绘制类别分布图"""
    if len(result.boxes) == 0:
        return None

    class_counts = {}
    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = CLASS_NAMES[cls_id]
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    fig, ax = plt.subplots(figsize=(10, 6))
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    sns.barplot(x=counts, y=classes, palette='viridis', ax=ax)
    ax.set_xlabel('检测数量', fontsize=12)
    ax.set_ylabel('类别', fontsize=12)
    ax.set_title('检测类别分布', fontsize=14, fontweight='bold')

    return fig


def main():
    st.set_page_config(
        page_title="YOLOv8 交通标志检测系统",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🚦 改进YOLOv8交通标志智能检测系统")
    st.caption("毕业设计项目 | 支持图片/视频/实时检测 | 多模型对比")
    st.markdown("---")

    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 系统设置")

        # 功能选择
        mode = st.selectbox(
            "选择功能模式",
            ["📷 单张图片检测", "🎥 视频检测", "📦 批量图片处理", "🔍 模型性能对比"]
        )

        st.markdown("---")

        # 模型选择
        if mode != "🔍 模型性能对比":
            selected_model = st.selectbox("选择模型", list(MODELS.keys()))
            model_path = MODELS[selected_model]
            st.info(f"当前模型: `{selected_model}`")

        st.markdown("---")

        # 检测参数
        st.subheader("检测参数")
        conf = st.slider("置信度阈值", 0.0, 1.0, 0.25, 0.05)
        iou = st.slider("IoU阈值", 0.0, 1.0, 0.45, 0.05)
        img_size = st.selectbox("推理尺寸", [320, 640, 1280], index=1)

        st.markdown("---")
        st.caption("💡 提示: 调整参数可优化检测效果")

    # ========== 单张图片检测 ==========
    if mode == "📷 单张图片检测":
        model = load_model(model_path)
        if model is None:
            st.stop()

        uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            col1, col2 = st.columns(2)

            image = Image.open(uploaded_file)

            with col1:
                st.subheader("原始图片")
                st.image(image, use_container_width=True)

            if st.button("🚀 开始检测", use_container_width=True):
                with st.spinner("检测中..."):
                    result = detect_image(model, image, conf, iou, img_size)

                with col2:
                    st.subheader("检测结果")
                    im_bgr = result.plot()
                    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
                    st.image(im_rgb, use_container_width=True)

                # 检测详情
                if len(result.boxes) > 0:
                    st.subheader("📊 检测详情")

                    tab1, tab2 = st.tabs(["检测列表", "类别分布"])

                    with tab1:
                        rows = []
                        for i, box in enumerate(result.boxes):
                            cls_id = int(box.cls[0])
                            conf_val = float(box.conf[0])
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                            rows.append({
                                "序号": i + 1,
                                "类别": CLASS_NAMES[cls_id],
                                "置信度": f"{conf_val:.2%}",
                                "坐标": f"({x1},{y1})-({x2},{y2})"
                            })

                        st.dataframe(pd.DataFrame(rows), use_container_width=True)

                    with tab2:
                        fig = plot_class_distribution(result)
                        if fig:
                            st.pyplot(fig)
                else:
                    st.info("⚠ 未检测到交通标志")

    # ========== 视频检测 ==========
    elif mode == "🎥 视频检测":
        model = load_model(model_path)
        if model is None:
            st.stop()

        uploaded_video = st.file_uploader("上传视频", type=["mp4", "avi", "mov"])

        if uploaded_video:
            # 保存临时文件
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())

            st.video(uploaded_video)

            if st.button("🎬 开始处理视频"):
                progress_bar = st.progress(0)

                with st.spinner("视频处理中,请耐心等待..."):
                    output_path, stats_df = detect_video(
                        model, tfile.name, conf, iou, img_size, progress_bar
                    )

                st.success("✅ 视频处理完成!")

                # 显示处理后的视频
                st.video(output_path)

                # 统计信息
                st.subheader("📈 检测统计")
                col1, col2, col3 = st.columns(3)
                col1.metric("总帧数", len(stats_df))
                col2.metric("平均检测数", f"{stats_df['detections'].mean():.1f}")
                col3.metric("最大检测数", stats_df['detections'].max())

    # ========== 批量处理 ==========
    elif mode == "📦 批量图片处理":
        model = load_model(model_path)
        if model is None:
            st.stop()

        uploaded_files = st.file_uploader(
            "上传多张图片",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        if uploaded_files:
            st.info(f"已上传 {len(uploaded_files)} 张图片")

            if st.button("📦 批量处理"):
                results_df = batch_process_images(
                    model, uploaded_files, conf, iou, img_size
                )

                st.subheader("处理结果汇总")
                st.dataframe(results_df, use_container_width=True)

                # 统计
                col1, col2 = st.columns(2)
                col1.metric("处理图片数", len(results_df))
                col2.metric("总检测数", results_df['检测数量'].sum())

    # ========== 模型对比 ==========
    elif mode == "🔍 模型性能对比":
        uploaded_file = st.file_uploader("上传测试图片", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file)

            st.subheader("原始图片")
            st.image(image, use_container_width=True)

            if st.button("🔍 对比所有模型"):
                st.subheader("模型检测结果对比")
                comparison_df = compare_models(image, MODELS, conf, iou, img_size)

                st.subheader("📊 性能对比表")
                st.dataframe(comparison_df, use_container_width=True)


if __name__ == "__main__":
    main()