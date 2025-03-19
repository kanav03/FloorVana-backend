#include "structure.h"
#pragma execution_character_set("utf-8")

QString English2Chinese(const QString &english)
{
	QString chinese = " ";
	if (english == "LivingRoom")
	{
		chinese = "����";
	}
	if (english == "MasterRoom")
	{
		chinese = "����";
	}
	if (english == "Kitchen")
	{
		chinese = "����";
	}
	if (english == "Bathroom")
	{
		chinese = "������";
	}
	if (english == "DiningRoom")
	{
		chinese = "�� ��";
	}
	if (english == "ChildRoom")
	{
		chinese = "��ͯ��";
	}
	if (english == "StudyRoom")
	{
		chinese = "�鷿";
	}
	if (english == "SecondRoom")
	{
		chinese = "����";
	}
	if (english == "GuestRoom")
	{
		chinese = "����";
	}
	if (english == "Balcony")
	{
		chinese = "��̨";
	}
	if (english == "Entrance")
	{
		chinese = "����";
	}
	if (english == "Storage")
	{ 
		chinese = "�����";
	}
	if (english == "Wall-in")
	{
		chinese = "�ڳ�";
	}

	return(chinese);
}