using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.ImageAnalytics;
using Microsoft.ML.Trainers;

namespace ImageClassification
{
    class Program
    {
        static readonly string _assetsPath = Path.Combine("..\\..\\..\\", "assets");
        //static readonly string _trainTagsTsv = Path.Combine(_assetsPath, "inputs-train", "data", "tags.tsv");
        static readonly string _trainTagsTsv = Path.Combine(_assetsPath, "inputs-train", "data", "tags2.tsv");
        static readonly string _predictImageListTsv = Path.Combine(_assetsPath, "inputs-predict", "data", "image_list.tsv");
        //static readonly string _trainImagesFolder = Path.Combine(_assetsPath, "inputs-train", "data");
        static readonly string _trainImagesFolder = Path.Combine(_assetsPath, "inputs-train", "Images");
        //static readonly string _trainImagesFolder = "D:\\Images";
        static readonly string _predictImagesFolder = Path.Combine(_assetsPath, "inputs-predict", "data");
        //static readonly string _predictSingleImage = Path.Combine(_assetsPath, "inputs-predict-single", "data", "toaster3.jpg");
        static readonly string _predictSingleImage = Path.Combine(_assetsPath, "inputs-predict-single", "data", "face7.jpg");
        static readonly string _inceptionPb = Path.Combine(_assetsPath, "inputs-train", "inception", "tensorflow_inception_graph.pb");
        static readonly string _inputImageClassifierZip = Path.Combine(_assetsPath, "inputs-predict", "imageClassifier.zip");
        static readonly string _outputImageClassifierZip = Path.Combine(_assetsPath, "outputs", "imageClassifier.zip");
        private static string LabelTokey = nameof(LabelTokey);
        private static string ImageReal = nameof(ImageReal);
        private static string PredictedLabelValue = nameof(PredictedLabelValue);

        static void Main(string[] args)
        {
            //初始化
            MLContext mlContext = new MLContext(seed: 1);

            //训练
            //ReuseAndTuneInceptionModel(mlContext, _trainTagsTsv, _trainImagesFolder, _inceptionPb, _outputImageClassifierZip);

            //评估
            //ClassifyImages(mlContext, _predictImageListTsv, _predictImagesFolder, _outputImageClassifierZip);

            //预测
            ClassifySingleImage(mlContext, _predictSingleImage, _outputImageClassifierZip);
        }

        private struct InceptionSettings
        {
            public const int ImageHeight = 224;
            public const int ImageWidth = 224;
            public const float Mean = 117;
            public const float Scale = 1;
            public const bool ChannelsLast = true;
        }

        public static void ReuseAndTuneInceptionModel(MLContext mlContext, string dataLocation, string imagesFolder, string inputModelLocation, string outputModelLocation)
        {
            //读取数据文件内容
            var data = mlContext.Data.ReadFromTextFile<ImageData>(path: dataLocation, hasHeader: false);

            var estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: LabelTokey, inputColumnName: DefaultColumnNames.Label)
                            //转换图像作为位图类型加载到内存中。
                            .Append(mlContext.Transforms.LoadImages(imagesFolder, (ImageReal, nameof(ImageData.ImagePath))))
                            //转换可重设图像大小，因为预定型模型有已定义的输入图像宽度和高度。
                            .Append(mlContext.Transforms.Resize(outputColumnName: ImageReal, imageWidth: InceptionSettings.ImageWidth, imageHeight: InceptionSettings.ImageHeight, inputColumnName: ImageReal))
                            //转换可提取输入图像中的像素，并将它们转换为数值向量。
                            .Append(mlContext.Transforms.ExtractPixels(new ImagePixelExtractorTransformer.ColumnInfo(name: "input", inputColumnName: ImageReal, interleave: InceptionSettings.ChannelsLast, offset: InceptionSettings.Mean)))

                            .Append(mlContext.Transforms.ScoreTensorFlowModel(modelLocation: inputModelLocation, outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }))
                            //添加定型算法
                            .Append(mlContext.MulticlassClassification.Trainers.LogisticRegression(labelColumn: LabelTokey, featureColumn: "softmax2_pre_activation"))
                            //将 predictedlabel 映射到 predictedlabelvalue
                            .Append(mlContext.Transforms.Conversion.MapKeyToValue((PredictedLabelValue, DefaultColumnNames.PredictedLabel)));

            // Train the model
            Console.WriteLine("=============== Training classification model ===============");

            ITransformer model = estimator.Fit(data);

            var predictions = model.Transform(data);

            var imageData = mlContext.CreateEnumerable<ImageData>(data, false, true);
            var imagePredictionData = mlContext.CreateEnumerable<ImagePrediction>(predictions, false, true);

            PairAndDisplayResults(imageData, imagePredictionData);

            Console.WriteLine("=============== Classification metrics ===============");

            var regressionContext = new MulticlassClassificationCatalog(mlContext);
            var metrics = regressionContext.Evaluate(predictions, label: LabelTokey, predictedLabel: DefaultColumnNames.PredictedLabel);

            Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
            Console.WriteLine($"PerClassLogLoss is: {String.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");

            Console.WriteLine("=============== Save model to local file ===============");

            using (var fileStream = new FileStream(outputModelLocation, FileMode.Create))
                mlContext.Model.Save(model, fileStream);

            Console.WriteLine($"Model saved: {outputModelLocation}");
        }

        public static void ClassifyImages(MLContext mlContext, string dataLocation, string imagesFolder, string outputModelLocation)
        {
            Console.WriteLine($"=============== Loading model ===============");
            Console.WriteLine($"Model loaded: {outputModelLocation}");

            ITransformer loadedModel;
            using (var fileStream = new FileStream(outputModelLocation, FileMode.Open))
                loadedModel = mlContext.Model.Load(fileStream);

            var imageData = ReadFromTsv(dataLocation, imagesFolder);
            var imageDataView = mlContext.Data.ReadFromEnumerable<ImageData>(imageData);

            var predictions = loadedModel.Transform(imageDataView);
            var imagePredictionData = mlContext.CreateEnumerable<ImagePrediction>(predictions, false, true);

            Console.WriteLine("=============== Making classifications ===============");

            PairAndDisplayResults(imageData, imagePredictionData);

        }

        public static void ClassifySingleImage(MLContext mlContext, string imagePath, string outputModelLocation)
        {
            Console.WriteLine($"=============== Loading model ===============");
            Console.WriteLine($"Model loaded: {outputModelLocation}");

            ITransformer loadedModel;
            using (var fileStream = new FileStream(outputModelLocation, FileMode.Open))
                loadedModel = mlContext.Model.Load(fileStream);

            var imageData = new ImageData()
            {
                ImagePath = imagePath
            };

            var predictor = loadedModel.CreatePredictionEngine<ImageData, ImagePrediction>(mlContext);
            var prediction = predictor.Predict(imageData);


            Console.WriteLine("=============== Making single image classification ===============");

            Console.WriteLine($"Image: {Path.GetFileName(imageData.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} ");


        }

        private static void PairAndDisplayResults(IEnumerable<ImageData> imageNetData, IEnumerable<ImagePrediction> imageNetPredictionData)
        {

            IEnumerable<(ImageData image, ImagePrediction prediction)> imagesAndPredictions = imageNetData.Zip(imageNetPredictionData, (image, prediction) => (image, prediction));

            foreach ((ImageData image, ImagePrediction prediction) item in imagesAndPredictions)
            {
                Console.WriteLine($"Image: {Path.GetFileName(item.image.ImagePath)} predicted as: {item.prediction.PredictedLabelValue} with score: {item.prediction.Score.Max()} ");
            }
        }

        public static IEnumerable<ImageData> ReadFromTsv(string file, string folder)
        {
            return File.ReadAllLines(file)
             .Select(line => line.Split('\t'))
             .Select(line => new ImageData()
             {
                 ImagePath = Path.Combine(folder, line[0])
             });
        }
    }
}
