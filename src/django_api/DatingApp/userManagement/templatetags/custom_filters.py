from django import template

register = template.Library()

@register.filter
def get(dictionary, key):
    return dictionary.get(key)

@register.filter
def is_list(value):
    return isinstance(value, list)