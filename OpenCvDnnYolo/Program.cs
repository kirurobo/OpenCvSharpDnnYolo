using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using OpenCvSharp;
using OpenCvSharp.Dnn;

namespace OpenCvDnnYolo
{
    class Program
    {
        //private static readonly string[] Labels = { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };
        private static readonly string[] Labels = {
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush"
        };

        private static readonly Scalar[] Colors = Enumerable.Repeat(false, 80).Select(x => Scalar.RandomColor()).ToArray();

        private static readonly string OutputDirectory = "..\\Output";

        static void Main(string[] args)
        {
            // https://pjreddie.com/darknet/yolo/
            //var cfg = "yolo-voc.cfg";
            //var model = "yolo-voc.weights"; //YOLOv2 544x544
            //var size = new Size(544, 544);
            //var cfg = "yolov3.cfg";
            //var model = "yolov3.weights"; //YOLOv2 544x544
            //var size = new Size(608, 608);
            //var cfg = "yolov3-tiny.cfg";
            //var model = "yolov3-tiny.weights";
            //var size = new Size(416, 416);
            var cfg = "yolov2.cfg";
            var model = "yolov2.weights";
            var net = CvDnn.ReadNetFromDarknet(cfg, model);

            string file;

            if (args.Length < 1)
            {
                //file = "bali.jpg";
                file = "dog.jpg";
                Process(net, file, true);
            }
            else if (args.Length == 1)
            {
                // 引数が画像1つの場合はウィンドウで画像表示

                file = args[0];
                LogError(file);
                if (File.Exists(file))
                {
                    Process(net, file, true);
                }
                else if (Directory.Exists(file))
                {
                    // フォルダならばその中を一括で処理
                    ProcessDirectory(net, file);

                    //Log("");
                    //Log("何かキーを押すと終了します");
                    //Console.ReadKey();
                }
            }
            else
            {
                // 複数の引数があった場合はウィンドウ表示はなし
                foreach (var path in args)
                {
                    if (File.Exists(path))
                    {
                        Process(net, path, false);
                    }
                    else
                    {
                        ProcessDirectory(net, path);
                    }
                }

                //Log("");
                //Log("何かキーを押すと終了します");
                //Console.ReadKey();
            }
        }

        static void ProcessDirectory(Net net, string dir)
        {
            Log("");
            Log("Retrieving " + dir);

            // サブディレクトリを再帰的に処理
            foreach (var path in Directory.EnumerateDirectories(dir))
            {
                ProcessDirectory(net, path);
            }

            // ディレクトリ中のファイルすべてを処理
            var files = Directory.EnumerateFiles(dir);
            var maxCount = files.Count();
            int count = 0;
            foreach (var path in files)
            {
                Process(net, path, false);
                count++;
                
                if (count % 100 == 0)
                {
                    Log("");
                    Log(count + "/" + maxCount);
                }
            }
        }

        static bool IsImageFile(string path)
        {
            string ext = Path.GetExtension(path).ToLower();
            switch (ext)
            {
                case ".jpg":    return true;
                case ".jpeg":   return true;
                case ".png":    return true;
                case ".gif":    return true;
            }
            return false;
        }

        /// <summary>
        /// ログ出力
        /// </summary>
        /// <param name="message"></param>
        static void Log(string message)
        {
            Console.WriteLine(message);
        }

        /// <summary>
        /// エラー出力
        /// </summary>
        /// <param name="message"></param>
        static void LogError(string message)
        {
            Console.WriteLine(message);
        }

        /// <summary>
        /// 1つ分ファイルを処理
        /// </summary>
        /// <param name="file"></param>
        /// <param name="show"></param>
        static void Process(Net net, string file, bool show = true)
        {
            if (!IsImageFile(file))
            {
                LogError(file + " is not an image.");
            }

            // https://pjreddie.com/darknet/yolo/
            //var cfg = "yolo-voc.cfg";
            //var model = "yolo-voc.weights"; //YOLOv2 544x544
            //var size = new Size(544, 544);
            //var cfg = "yolov3.cfg";
            //var model = "yolov3.weights"; //YOLOv2 544x544
            //var size = new Size(608, 608);
            //var cfg = "yolov3-tiny.cfg";
            //var model = "yolov3-tiny.weights";
            //var size = new Size(416, 416);
            var size = new Size(608, 608);

            var threshold = 0.23;
            //var threshold = 0.1;

            var org = Cv2.ImRead(file);
            var w = org.Width;
            var h = org.Height;
            //setting blob, parameter are important
            var blob = CvDnn.BlobFromImage(org, 1 / 255.0, size, new Scalar(), true, false);
            //var blob = CvDnn.BlobFromImage(org, 1 / 255.0, size);
            net.SetInput(blob, "data");

            Mat prob;

            if (show)
            {
                Stopwatch sw = new Stopwatch();
                sw.Start();
                //forward model
                prob = net.Forward();
                sw.Stop();
                Log($"Runtime:{sw.ElapsedMilliseconds} ms");
            }
            else
            {
                try
                {
                    prob = net.Forward();
                    Console.Write(".");
                }
                catch
                {
                    Console.WriteLine("");
                    LogError("ERROR classification: " + file);
                    return;
                }
            }

            /* YOLO2 VOC output
             0 1 : center                    2 3 : w/h
             4 : confidence                  5 ~24 : class probability */
            const int prefix = 5;   //skip 0~4

            int selectedIndex = -1; // 検出された物体のうち、面積が最大となるものの番号
            float maxArea = 0f;
            string selectedLabel = "";

            for (int i = 0; i < prob.Rows; i++)
            {
                var confidence = prob.At<float>(i, 4);
                if (confidence > threshold)
                {
                    //get classes probability
                    Cv2.MinMaxLoc(prob.Row[i].ColRange(prefix, prob.Cols), out _, out Point max);
                    var classes = max.X;
                    var probability = prob.At<float>(i, classes + prefix);

                    if (probability > threshold) //more accuracy
                    {
                        //get center and width/height
                        var centerX = prob.At<float>(i, 0) * w;
                        var centerY = prob.At<float>(i, 1) * h;
                        var width = prob.At<float>(i, 2) * w;
                        var height = prob.At<float>(i, 3) * h;

                        var area = width * height;
                        if (area > maxArea)
                        {
                            selectedIndex = i;
                            selectedLabel = Labels[classes];
                            maxArea = area;
                        }

                        if (show)
                        {
                            //label formating
                            var label = $"{Labels[classes]} {probability * 100:0.00}%";
                            Log($"confidence {confidence * 100:0.00}% {label}");
                            var x1 = (centerX - width / 2) < 0 ? 0 : centerX - width / 2; //avoid left side over edge
                                                                                          //draw result
                            org.Rectangle(new Point(x1, centerY - height / 2), new Point(centerX + width / 2, centerY + height / 2), Colors[classes], 2);
                            var textSize = Cv2.GetTextSize(label, HersheyFonts.HersheyTriplex, 0.5, 1, out var baseline);
                            Cv2.Rectangle(org, new Rect(new Point(x1, centerY - height / 2 - textSize.Height - baseline),
                                    new Size(textSize.Width, textSize.Height + baseline)), Colors[classes], Cv2.FILLED);
                            Cv2.PutText(org, label, new Point(x1, centerY - height / 2 - baseline), HersheyFonts.HersheyTriplex, 0.5, Scalar.Black);

                            //Console.WriteLine("{0}, {1:0.00}%" , label, confidence * 100.0);
                        }
                    }
                }
            }

            // 検出物体があったときの処理
            if (selectedIndex >= 0 && !show)
            {
                // ラベル名のディレクトリに画像を移動
                var newDir = Path.Combine(OutputDirectory, selectedLabel);
                Directory.CreateDirectory(newDir);

                try
                {
                    File.Move(file, Path.Combine(newDir, Path.GetFileName(file)));
                }
                catch
                {
                    Console.WriteLine("");
                    LogError("ERROR move: " + file);
                }
                //Log(Directory.GetCurrentDirectory());
                //Log(Path.Combine(newDir, Path.GetFileName(file)));
            }

            if (show)
            {
                using (new Window("died.tw", org))
                {
                    Cv2.WaitKey();
                }
            }
        }
    }
}
