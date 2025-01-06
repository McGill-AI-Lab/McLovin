# DatingApp

DatingApp is a web application built with Django as the backend framework and MongoDB as the database. This app includes essential dating features like user authentication, data input/output, and a dynamic landing page.

---

## Features

- **Authentication**: Secure user login and registration.
- **Landing Page**: A dynamic and interactive homepage.
- **Data Input/Output**: Allows users to input and retrieve matchmaking data.

---

## Tech Stack

### Backend
- **Framework**: [Django](https://www.djangoproject.com/)
  - Batteries-included framework with built-in libraries.
  - Includes Object-Relational Mapping (ORM) for seamless database interaction.

### Database
- **MongoDB**
  - Non-relational database optimized for scalability and flexibility.

### Frontend
- Compatible with `pytorch` for machine learning integration.
- Ensure dependencies are updated using `npm install` after pulling the repository.

---

## Project Structure

```
└── DatingApp
    ├── DatingApp
    │   ├── __init__.py
    │   ├── asgi.py
    │   ├── settings.py
    │   ├── urls.py
    │   └── wsgi.py
    ├── databaseInteraction
    │   ├── __init__.py
    │   ├── admin.py
    │   ├── apps.py
    │   ├── models.py
    │   ├── tests.py
    │   └── views.py
    ├── db.sqlite3
    ├── home
    │   ├── __init__.py
    │   ├── admin.py
    │   ├── apps.py
    │   ├── models.py
    │   ├── templates
    │   │   ├── base.html
    │   │   ├── child.html
    │   │   ├── home.html
    │   │   ├── imported.html
    │   │   ├── module1.html
    │   │   └── mongotry.py
    │   ├── tests.py
    │   └── views.py
    ├── manage.py
    └── templates
```

2. Create a virtual environment for python execution to avoid conflicts with your global install 
> python -m venv venv
> source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

4. install front depedencies (cd to folder with .json  config file)
> npm install

5. Run the development server with 
`python manage.py runserver` with workdir `DatingApp` folder that contains the manage.py file 
