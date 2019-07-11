using Microsoft.ML.Data;

namespace LinearRegressionMLNet
{
    class BikeShare
    {
        [LoadColumn(0)]
        public float instant;

        [LoadColumn(1)]
        public float dteday;

        [LoadColumn(2)]
        public float season;

        [LoadColumn(3)]
        public float yr;

        [LoadColumn(4)]
        public float mnth;

        [LoadColumn(5)]
        public float holiday;

        [LoadColumn(6)]
        public float weekday;

        [LoadColumn(7)]
        public float workingday;

        [LoadColumn(8)]
        public float weathersit;

        [LoadColumn(9)]
        public float temp;

        [LoadColumn(10)]
        public float atemp;

        [LoadColumn(11)]
        public float hum;

        [LoadColumn(12)]
        public float windspeed;

        [LoadColumn(13)]
        public float casual;

        [LoadColumn(14)]
        public float registered;

        [LoadColumn(15)]
        public float cnt;
    }
}
