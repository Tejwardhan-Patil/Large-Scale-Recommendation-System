import slick.jdbc.PostgresProfile.api._
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.concurrent.duration._
import java.time.LocalDateTime
import scala.util.{Failure, Success}

// Define the main object
object DatabaseSetup {

  // Define the ExecutionContext
  implicit val ec: ExecutionContext = ExecutionContext.global

  // Logging utilities
  def logMessage(message: String): Unit = {
    val timestamp = LocalDateTime.now
    println(s"[$timestamp] - $message")
  }

  // Define the users table schema
  class Users(tag: Tag) extends Table[(Int, String, String, String, Option[LocalDateTime], Option[LocalDateTime], Option[LocalDateTime], String)](tag, "users") {
    def userId = column[Int]("user_id", O.PrimaryKey, O.AutoInc)
    def username = column[String]("username", O.Unique)
    def email = column[String]("email", O.Unique)
    def passwordHash = column[String]("password_hash")
    def createdAt = column[Option[LocalDateTime]]("created_at", O.Default(Some(LocalDateTime.now)))
    def updatedAt = column[Option[LocalDateTime]]("updated_at")
    def lastLogin = column[Option[LocalDateTime]]("last_login")
    def status = column[String]("status", O.Default("active"))

    def * = (userId, username, email, passwordHash, createdAt, updatedAt, lastLogin, status)
  }

  // Define the items table schema
  class Items(tag: Tag) extends Table[(Int, String, String, Int, BigDecimal, Int, Option[LocalDateTime], Option[LocalDateTime], String)](tag, "items") {
    def itemId = column[Int]("item_id", O.PrimaryKey, O.AutoInc)
    def itemName = column[String]("item_name")
    def itemDescription = column[String]("item_description")
    def categoryId = column[Int]("category_id")
    def price = column[BigDecimal]("price")
    def stockQuantity = column[Int]("stock_quantity", O.Default(0))
    def createdAt = column[Option[LocalDateTime]]("created_at", O.Default(Some(LocalDateTime.now)))
    def updatedAt = column[Option[LocalDateTime]]("updated_at")
    def status = column[String]("status", O.Default("available"))

    def * = (itemId, itemName, itemDescription, categoryId, price, stockQuantity, createdAt, updatedAt, status)

    // Foreign key relationship
    def category = foreignKey("fk_category", categoryId, categoriesTable)(_.categoryId, onDelete = ForeignKeyAction.Cascade)
  }

  // Define the categories table schema
  class Categories(tag: Tag) extends Table[(Int, String, Option[Int], Option[LocalDateTime], Option[LocalDateTime])](tag, "categories") {
    def categoryId = column[Int]("category_id", O.PrimaryKey, O.AutoInc)
    def categoryName = column[String]("category_name", O.Unique)
    def parentCategoryId = column[Option[Int]]("parent_category_id")
    def createdAt = column[Option[LocalDateTime]]("created_at", O.Default(Some(LocalDateTime.now)))
    def updatedAt = column[Option[LocalDateTime]]("updated_at")

    def * = (categoryId, categoryName, parentCategoryId, createdAt, updatedAt)

    def parentCategory = foreignKey("fk_parent_category", parentCategoryId, categoriesTable)(_.categoryId, onDelete = ForeignKeyAction.SetNull)
  }

  // Define the user_roles table schema
  class UserRoles(tag: Tag) extends Table[(Int, String, Option[LocalDateTime], Option[LocalDateTime])](tag, "user_roles") {
    def roleId = column[Int]("role_id", O.PrimaryKey, O.AutoInc)
    def roleName = column[String]("role_name", O.Unique)
    def createdAt = column[Option[LocalDateTime]]("created_at", O.Default(Some(LocalDateTime.now)))
    def updatedAt = column[Option[LocalDateTime]]("updated_at")

    def * = (roleId, roleName, createdAt, updatedAt)
  }

  // Define the user_role_mapping table schema
  class UserRoleMapping(tag: Tag) extends Table[(Int, Int, Option[LocalDateTime])](tag, "user_role_mapping") {
    def userId = column[Int]("user_id")
    def roleId = column[Int]("role_id")
    def assignedAt = column[Option[LocalDateTime]]("assigned_at", O.Default(Some(LocalDateTime.now)))

    def * = (userId, roleId, assignedAt)

    def user = foreignKey("fk_user", userId, usersTable)(_.userId, onDelete = ForeignKeyAction.Cascade)
    def role = foreignKey("fk_role", roleId, userRolesTable)(_.roleId, onDelete = ForeignKeyAction.Cascade)
    def pk = primaryKey("pk_user_role", (userId, roleId))
  }

  // Define the item_views table schema
  class ItemViews(tag: Tag) extends Table[(Int, Int, Int, LocalDateTime)](tag, "item_views") {
    def viewId = column[Int]("view_id", O.PrimaryKey, O.AutoInc)
    def itemId = column[Int]("item_id")
    def userId = column[Int]("user_id")
    def viewedAt = column[LocalDateTime]("viewed_at", O.Default(LocalDateTime.now))

    def * = (viewId, itemId, userId, viewedAt)

    def item = foreignKey("fk_item", itemId, itemsTable)(_.itemId, onDelete = ForeignKeyAction.Cascade)
    def user = foreignKey("fk_user", userId, usersTable)(_.userId, onDelete = ForeignKeyAction.Cascade)
  }

  // Instantiate table queries
  val usersTable = TableQuery[Users]
  val itemsTable = TableQuery[Items]
  val categoriesTable = TableQuery[Categories]
  val userRolesTable = TableQuery[UserRoles]
  val userRoleMappingTable = TableQuery[UserRoleMapping]
  val itemViewsTable = TableQuery[ItemViews]

  // Database setup
  val db = Database.forConfig("mydb")

  def main(args: Array[String]): Unit = {
    try {
      // Define the schema creation and initial data seeding
      val setupAction = DBIO.seq(
        // Create tables in order to handle foreign key dependencies
        categoriesTable.schema.createIfNotExists,
        usersTable.schema.createIfNotExists,
        userRolesTable.schema.createIfNotExists,
        itemsTable.schema.createIfNotExists,
        userRoleMappingTable.schema.createIfNotExists,
        itemViewsTable.schema.createIfNotExists,

        // Seed initial categories data
        categoriesTable ++= Seq(
          (1, "Electronics", None, Some(LocalDateTime.now), None),
          (2, "Books", None, Some(LocalDateTime.now), None),
          (3, "Clothing", None, Some(LocalDateTime.now), None)
        ),

        // Seed initial roles data
        userRolesTable ++= Seq(
          (1, "Admin", Some(LocalDateTime.now), None),
          (2, "User", Some(LocalDateTime.now), None)
        ),

        // Seed initial users
        usersTable ++= Seq(
          (1, "admin", "admin@website.com", "hashed_password", Some(LocalDateTime.now), None, None, "active"),
          (2, "max", "maxe@website.com", "hashed_password", Some(LocalDateTime.now), None, None, "active")
        ),

        // Assign roles to users
        userRoleMappingTable ++= Seq(
          (1, 1, Some(LocalDateTime.now)), // Admin role for admin user
          (2, 2, Some(LocalDateTime.now))  // User role for max
        )
      )

      // Running migration and seeding with detailed logging
      logMessage("Starting migration and data seeding...")

      val setupFuture: Future[Unit] = db.run(setupAction)

      setupFuture.onComplete {
        case Success(_) =>
          logMessage("Migration and data seeding completed successfully.")
          // Close the database connection
          db.close()
          logMessage("Database connection closed.")
        case Failure(ex) =>
          logMessage(s"Migration failed: ${ex.getMessage}")
          // Close the DB even on failure
          db.close()
          logMessage("Database connection closed due to failure.")
      }

      // Await the result for a specified duration
      Await.result(setupFuture, 20.minutes)
    } catch {
      case ex: Exception =>
        logMessage(s"An unexpected error occurred: ${ex.getMessage}")
        db.close()
        logMessage("Database connection closed due to unexpected error.")
    }
  }
}