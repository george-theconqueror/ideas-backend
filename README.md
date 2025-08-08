# Ideas Backend API

A FastAPI backend that manages a ranking system for challenge ideas using Elo ratings.

## Features

- **Random Titles**: Get shuffled random titles from the database
- **Elo Rating System**: Update ratings based on head-to-head comparisons

## API Endpoints

- `GET /random-titles` - Get 50 random titles with their ratings
- `POST /update-elo` - Update Elo ratings after a comparison

## Setup

1. Create `.env` file with `DATABASE_URL`
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `uvicorn main:app --reload`

