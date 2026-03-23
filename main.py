from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import database_models
from database import SessionLocal, engine
from models import Product, ProductCreate

database_models.Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

sample_products = [
    ProductCreate(name="Phone",  description="A smartphone",      price=699.99, quantity=50),
    ProductCreate(name="Laptop", description="A powerful laptop", price=999.99, quantity=30),
    ProductCreate(name="Pen",    description="A blue ink pen",    price=1.99,   quantity=100),
    ProductCreate(name="Table",  description="A wooden table",    price=199.99, quantity=20),
]

def init_db():
    db = SessionLocal()
    try:
        existing_count = db.query(database_models.Product).count()
        if existing_count == 0:
            for product in sample_products:
                db.add(database_models.Product(**product.model_dump()))
            db.commit()
            print("Database initialized with sample products.")
    finally:
        db.close()

init_db()

@app.get("/")
def root():
    return {"message": "Welcome to the Products API"}

@app.get("/products/")
def get_all_products(db: Session = Depends(get_db)):
    return db.query(database_models.Product).all()

@app.get("/products/{product_id}")
def get_product_by_id(product_id: int, db: Session = Depends(get_db)):
    product = db.query(database_models.Product).filter(
        database_models.Product.id == product_id
    ).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product

@app.post("/products/")
def create_product(product: ProductCreate, db: Session = Depends(get_db)):
    db_product = database_models.Product(**product.model_dump())
    db.add(db_product)
    db.commit()
    db.refresh(db_product)
    return {"message": "Product created successfully", "product": db_product}

@app.put("/products/{product_id}")
def update_product(product_id: int, product: ProductCreate, db: Session = Depends(get_db)):
    db_product = db.query(database_models.Product).filter(
        database_models.Product.id == product_id
    ).first()
    if not db_product:
        raise HTTPException(status_code=404, detail="Product not found")
    db_product.name        = product.name
    db_product.description = product.description
    db_product.price       = product.price
    db_product.quantity    = product.quantity
    db.commit()
    db.refresh(db_product)
    return {"message": "Product updated successfully", "product": db_product}

@app.delete("/products/{product_id}")
def delete_product(product_id: int, db: Session = Depends(get_db)):
    db_product = db.query(database_models.Product).filter(
        database_models.Product.id == product_id
    ).first()
    if not db_product:
        raise HTTPException(status_code=404, detail="Product not found")
    db.delete(db_product)
    db.commit()
    return {"message": "Product deleted successfully"}