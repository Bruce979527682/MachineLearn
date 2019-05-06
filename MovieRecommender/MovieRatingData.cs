﻿using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace MovieRecommender
{
    // <SnippetMovieRatingClass>
    public class MovieRating
    {
        [LoadColumn(0)]
        public float userId;
        [LoadColumn(1)]
        public float movieId;
        [LoadColumn(2)]
        public float Label;
    }
    // </SnippetMovieRatingClass>

    // <SnippetPredictionClass>
    public class MovieRatingPrediction
    {
        public float Label;
        public float Score;
    }
    // </SnippetPredictionClass>
}
