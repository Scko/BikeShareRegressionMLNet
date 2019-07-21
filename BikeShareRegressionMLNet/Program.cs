using Microsoft.ML;
using System;
using System.IO;

namespace LinearRegressionMLNet
{
    class Program
    {
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "bkTrain.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "bkTest.csv");

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 0);

            var model = Train<BikeShare>(mlContext, _trainDataPath);

            Evaluate<BikeShare>(mlContext, model);

            TestPrediction(mlContext, model);
        }

        public static ITransformer Train<T>(MLContext mlContext, string dataPath)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<T>(dataPath, hasHeader: true, separatorChar: ',');

            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "cnt")
                    .Append(mlContext.Transforms.Concatenate("Features", "season", "yr", "mnth", "holiday", "workingday", "weathersit", "temp", "hum", "windspeed"))
                    .Append(mlContext.Regression.Trainers.FastTree());

            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = pipeline.Fit(dataView);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();

            return model;
        }

        private static void Evaluate<T>(MLContext mlContext, ITransformer model)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<T>(_testDataPath, hasHeader: true, separatorChar: ',');
            var predictions = model.Transform(dataView);

            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
            Console.WriteLine($"*************************************************");
        }

        private static void TestPrediction(MLContext mlContext, ITransformer model)
        {
            //Prediction test
            // Create prediction function and make prediction.
            var predictionFunction = mlContext.Model.CreatePredictionEngine<BikeShare, BikeSharePrediction>(model);
            var sample = new BikeShare()
            {
                season = 1,
                yr = 1,
                mnth = 12,
                holiday = 0,
                weekday = 1,
                workingday = 1,
                weathersit = 2,
                temp = .215833f,
                atemp = .223487f,
                hum = .5775f,
                windspeed = .154846f,
                cnt = 0 // To predict. Actual/Observed = 2729
            };

            var sample2 = new BikeShare()
            {
                season = 3,
                yr = 1,
                mnth = 8,
                holiday = 0,
                weekday = 1,
                workingday = 1,
                weathersit = 2,
                temp = 0.7525f,
                atemp = 0.710246f,
                hum = 0.654167f,
                windspeed = 0.129354f,
                cnt = 0 // To predict. Actual/Observed = 7013
            };


            var prediction = predictionFunction.Predict(sample);
            var prediction2 = predictionFunction.Predict(sample2);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted cnt: {prediction.cnt:0.####}, actual cnt: 2729");
            Console.WriteLine($"Predicted cnt: {prediction2.cnt:0.####}, actual cnt: 7013");
            Console.WriteLine($"**********************************************************************");
        }
    }
}