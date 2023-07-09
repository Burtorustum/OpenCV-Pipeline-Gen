import streamlit as st


# TODO: maybe java pipelines should use a builder template and the library can have a builder class w all the methods built in

def generate_java(pipeline, selected_stages):
    import_code = ("import java.util.ArrayList;\n"
                   "import java.util.List;\n"
                   "import org.opencv.core.Core;\n"
                   "import org.opencv.core.Mat;\n"
                   "import org.opencv.core.MatOfPoint;\n"
                   "import org.opencv.core.Point;\n"
                   "import org.opencv.core.Rect;\n"
                   "import org.opencv.core.Scalar;\n"
                   "import org.opencv.core.Size;\n"
                   "import org.opencv.imgproc.Imgproc;\n"
                   "import org.openftc.easyopencv.OpenCvPipeline;\n"
                   )

    class_code = ("\n"
                  "public class Pipeline extends OpenCvPipeline {\n\n"
                  "    public Mat cameraIn;\n")

    process_frame_code = ("\n"
                          "    @Override\n"
                          "    public Mat processFrame(Mat input) {\n"
                          "        cameraIn = input;\n")

    post_code = """}\n"""

    methods = []
    last_output = "input"

    if "Crop" in selected_stages:
        methods.append(gen_crop())
        process_frame_code += f"        applyCrop({last_output});\n"
        class_code += "    public Mat cropOut = new Mat();\n"
        last_output = "cropOut"

    for stage in pipeline:
        match stage:
            case "Blur":
                methods.append(gen_blur())
                process_frame_code += f"        applyBlur({last_output});\n"
                class_code += "    public final Mat blurOut = new Mat();\n"
                last_output = "blurOut"

            case "Erode":
                methods.append(gen_erode())
                process_frame_code += f"        applyErode({last_output});\n"
                class_code += "    public final Mat erodeOut = new Mat();\n"
                last_output = "erodeOut"

            case "Dilate":
                methods.append(gen_dilate())
                process_frame_code += f"        applyDilate({last_output});\n"
                class_code += "    public final Mat dilateOut = new Mat();\n"
                last_output = "dilateOut"

            case "Threshold":
                methods.append(gen_threshold())
                process_frame_code += f"        applyThreshold({last_output});\n"
                class_code += ("    public final Mat cvtColorOut = new Mat();\n"
                               "    public final Mat thresholdOut = new Mat();\n")
                last_output = "thresholdOut"

            case "Contours":
                methods.append(gen_contours())
                process_frame_code += f"        findContours({last_output});\n"
                class_code += ("    public final List<MatOfPoint> contours = new ArrayList<>();\n"
                               "    public final Mat contourHierarchy = new Mat();\n")

    process_frame_code += ("\n        return cameraIn;\n"
                           "    }\n")

    java_methods = ""
    for m in methods:
        java_methods += m

    st.code(body=import_code + class_code + process_frame_code + java_methods + post_code,
            language="java",
            line_numbers=True)


def gen_crop():
    left, top, width, height = tuple(map(int, st.session_state["crop_rect"].values()))
    return f"""
    private void applyCrop(Mat input) {{
        Rect cropRectangle = new Rect({left}, {top}, {width}, {height});
        this.cropOut = new Mat(input, cropRectangle);
    }}\n"""


def gen_blur():
    return f"""
    private void applyBlur(Mat input) {{
        int blurWidth = {st.session_state["blur_width"]};
        int blurHeight = {st.session_state["blur_height"]};
        Imgproc.blur(input, blurOut, new Size(blurWidth, blurHeight));
    }}\n"""


def match_shape_string(shape):
    match shape:
        case "RECT":
            return "Imgproc.MORPH_RECT"
        case "ELLIPSE":
            return "Imgproc.MORPH_ELLIPSE"
        case "CROSS":
            return "Imgproc.MORPH_CROSS"
        case _:
            return None


def gen_erode():
    return f"""
    private void applyErode(Mat input) {{
        int erodeSize = {st.session_state["erode_size"]};
        int erodeShape = {match_shape_string(st.session_state["erode_shape"])};
        Size erodeKernel = new Size(2 * erodeSize + 1, 2 * erodeSize + 1);
        Point erodeAnchor = new Point(erodeSize, erodeSize);
        int erodeIterations = {st.session_state["erode_iter"]};
    
        Imgproc.erode(
            input, 
            erodeOut, 
            Imgproc.getStructuringElement(erodeShape, erodeKernel),
            erodeAnchor,
            erodeIterations
        );
    }}\n"""


def gen_dilate():
    return f"""
    private void applyDilate(Mat input) {{
        int dilateSize = {st.session_state["dilate_size"]};
        int dilateShape = {match_shape_string(st.session_state["dilate_shape"])};
        Size dilateKernel = new Size(2 * dilateSize + 1, 2 * dilateSize + 1);
        Point dilateAnchor = new Point(dilateSize, dilateSize);
        int dilateIterations = {st.session_state["dilate_iter"]};
    
        Imgproc.dilate(
            input, 
            dilateOut, 
            Imgproc.getStructuringElement(dilateShape, dilateKernel),
            dilateAnchor,
            dilateIterations
        );
    }}\n"""


def gen_threshold():
    return f"""
    private void applyThreshold(Mat input) {{
        double[] hue = {str(st.session_state["hue"]).replace("(", "{").replace(")", "}")};
        double[] sat = {str(st.session_state["sat"]).replace("(", "{").replace(")", "}")};
        double[] val = {str(st.session_state["val"]).replace("(", "{").replace(")", "}")};
        boolean invert = {st.session_state["thresh_invert"]};
        
        // EasyOpenCV delivers RGBA frames, not BGR like normal OpenCV
        Imgproc.cvtColor(input, cvtColorOut, Imgproc.COLOR_RGBA2RGB);
        Imgproc.cvtColor(cvtColorOut, cvtColorOut, Imgproc.COLOR_RGB2HSV);
        Core.inRange(
            cvtColorOut, 
            new Scalar(hue[0], sat[0], val[0]),
            new Scalar(hue[1], sat[1], val[1]),
            thresholdOut
        );
        if (invert) {{ 
            Core.bitwise_not(thresholdOut, thresholdOut);
        }}
    }}\n"""


def match_approx_method_string(approx_method):
    match approx_method:
        case "CHAIN NONE":
            return "Imgproc.CHAIN_APPROX_NONE"
        case "CHAIN SIMPLE":
            return "Imgproc.CHAIN_APPROX_SIMPLE"
        case "CHAIN TC89 L1":
            return "Imgproc.CHAIN_APPROX_TC89_L1"
        case "CHAIN TC89 KCOS":
            return "Imgproc.CHAIN_APPROX_TC89_KCOS"
        case _:
            return None


def gen_contours():
    return f"""
    private void findContours(Mat input) {{
        Imgproc.findContours(input, contours, contourHierarchy, Imgproc.RETR_EXTERNAL, {match_approx_method_string(st.session_state["contour_approx_method"])});
        Imgproc.drawContours(cameraIn, contours, -1, new Scalar(0,255,0), 3);
    }}\n"""
