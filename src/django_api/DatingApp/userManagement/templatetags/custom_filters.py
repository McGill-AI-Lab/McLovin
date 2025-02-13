from django import template

register = template.Library()

enum_fields = {
    "ethnicity",
    "genders_of_interest",
    "major",
}
#    "faculty"
#    "grade"
#    "gender",

@register.filter
def get(dictionary, key):
    return dictionary.get(key)

@register.filter
def is_list(value):
    return isinstance(value, list)

@register.filter
def is_dropdown(value):
    print("FILTER RESULT WITH",value,value in enum_fields)
    return value in enum_fields