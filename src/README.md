### current configuration : 

* Django projects are structured around apps (here : `databaseInteraction`,`home`)
	* `home` has been created for testing dynamic templating features of Django
	* `databaseInteraction` has been created for testing database support with django (mongodb)

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
