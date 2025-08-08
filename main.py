from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from pydantic import BaseModel
import os

# Load environment variables from .env file
load_dotenv()

# Get database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

# Pydantic models
class GameResult(BaseModel):
    title1_id: int
    title2_id: int
    winner: int  # 0 for title1, 1 for title2

# Simple database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://the-conqueror-challenge-ideas.vercel.app"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=4)

# Get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



# GET endpoint to retrieve 50 random titles
@app.get("/random-titles")
async def get_random_titles(db: Session = Depends(get_db)):
    """Retrieve 50 random titles from the 'titles' table and return them shuffled"""
    
    result = db.execute(
        text("SELECT * FROM titles ORDER BY RANDOM() LIMIT 50")
    )
    
    titles = [{"id": row[0], "title": row[1], "elo": row[2], "appearances": row[3]} for row in result.fetchall()]
    
    # Shuffle the array
    import random
    random.shuffle(titles)
    
    return {"titles": titles, "count": len(titles)}

def calculate_expected_score(rating1: int, rating2: int) -> float:
    """Calculate expected score for player 1"""
    return 1 / (1 + 10**((rating2 - rating1) / 400))

def get_k_factor(appearances: int, elo: int) -> int:
    """Determine K-factor based on experience and rating (starting elo = 100)"""
    if appearances < 20:
        return 40  
    elif elo >= 600:  
        return 10  
    else:
        return 20  

@app.post("/update-elo")
async def update_elo(game_result: GameResult, db: Session = Depends(get_db)):
    """
    Update Elo ratings for two titles based on game result.
    Uses SELECT FOR UPDATE to prevent race conditions.
    """
    
    # Validate that the two IDs are different
    if game_result.title1_id == game_result.title2_id:
        raise HTTPException(status_code=400, detail="Cannot play against the same title")
    
    try:
        db.begin()
        
        # Lock both titles in consistent order (lower ID first) to prevent deadlocks
        id1, id2 = sorted([game_result.title1_id, game_result.title2_id])
        
        # Fetch both titles with row-level locking
        query = text("""
            SELECT id, title, elo, appearances 
            FROM titles 
            WHERE id IN (:id1, :id2)
            ORDER BY id
            FOR UPDATE
        """)
        
        result = db.execute(query, {"id1": id1, "id2": id2}).fetchall()
        
        if len(result) != 2:
            raise HTTPException(status_code=404, detail="One or both titles not found")
        
        # Map results back to original order
        titles = {}
        for row in result:
            titles[row[0]] = {
                'id': int(row[0]),
                'title': row[1],
                'elo': int(row[2]),
                'appearances': int(row[3])
            }
        
        title1 = titles[game_result.title1_id]
        title2 = titles[game_result.title2_id]
        
        # Calculate expected scores
        expected1 = calculate_expected_score(title1['elo'], title2['elo'])
        expected2 = 1 - expected1
        
        # Determine actual scores based on winner
        if game_result.winner == 0:  # title1 won
            actual1, actual2 = 1.0, 0.0
        else:  # title2 won
            actual1, actual2 = 0.0, 1.0
        
        # Get K-factors
        k1 = get_k_factor(title1['appearances'], title1['elo'])
        k2 = get_k_factor(title2['appearances'], title2['elo'])
        
        # Calculate rating changes
        delta1 = k1 * (actual1 - expected1)
        delta2 = k2 * (actual2 - expected2)
        
        # Calculate new ratings (ensure minimum rating of 100)
        new_elo1 = max(100, title1['elo'] + round(delta1))
        new_elo2 = max(100, title2['elo'] + round(delta2))
        
        # Update both titles
        update_query = text("""
            UPDATE titles 
            SET elo = :new_elo, appearances = appearances + 1
            WHERE id = :title_id
        """)
        
        db.execute(update_query, {"new_elo": new_elo1, "title_id": title1['id']})
        db.execute(update_query, {"new_elo": new_elo2, "title_id": title2['id']})
        
        # Commit transaction
        db.commit()
        
        # Return the results
        return {
            "success": True,
            "title1": {
                "id": title1['id'],
                "title": title1['title'],
                "old_elo": title1['elo'],
                "new_elo": new_elo1,
                "change": round(delta1),
                "appearances": title1['appearances'] + 1
            },
            "title2": {
                "id": title2['id'], 
                "title": title2['title'],
                "old_elo": title2['elo'],
                "new_elo": new_elo2,
                "change": round(delta2),
                "appearances": title2['appearances'] + 1
            },
            "expected_scores": {
                "title1": round(expected1, 3),
                "title2": round(expected2, 3)
            }
        }
        
    except Exception as e:
        # Rollback on any error
        db.rollback()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)