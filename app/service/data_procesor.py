from collections import Counter
from app.database import sales_collection


def inspect_dataset():

    total_sales = sales_collection.count_documents({})

    pipeline = [
        {
            "$match": {
                "fechaVenta": {"$type": "date"}
            }
        },
        {
            "$group": {
                "_id": None,
                "fecha_minima": {"$min": "$fechaVenta"},
                "fecha_maxima": {"$max": "$fechaVenta"},
                "total_con_fecha": {"$sum": 1}
            }
        }
    ]

    date_result = list(sales_collection.aggregate(pipeline))

    print("\n========== HISTÓRICO DE VENTAS ==========")
    print("Total de ventas:", total_sales)

    if date_result:
        print("Primera venta:", date_result[0]["fecha_minima"])
        print("Última venta:", date_result[0]["fecha_maxima"])
        print("Ventas con fecha:", date_result[0]["total_con_fecha"])

    # Consultar únicamente los campos necesarios
    sales = sales_collection.find(
        {},
        {"fechaVenta": 1, "helados": 1}
    )

    flavors = Counter()
    product_records = 0
    sales_without_date = 0
    sales_without_products = 0

    for sale in sales:

        if not sale.get("fechaVenta"):
            sales_without_date += 1

        products = sale.get("helados", [])

        if not products:
            sales_without_products += 1

        for item in products:
            helado = item.get("helado", {})

            flavor = helado.get("sabor")
            quantity = item.get("cantidad", 0)

            if flavor and isinstance(quantity, (int, float)):
                flavors[flavor.lower()] += quantity
                product_records += 1

    print("\n========== PRODUCTOS ==========")
    print("Registros de productos:", product_records)
    print("Sabores diferentes:", len(flavors))

    print("\n========== UNIDADES POR SABOR ==========")

    for flavor, quantity in flavors.most_common():
        print(f"{flavor}: {quantity}")

    print("\n========== VALIDACIÓN ==========")
    print("Ventas sin fecha:", sales_without_date)
    print("Ventas sin productos:", sales_without_products)


if __name__ == "__main__":
    inspect_dataset()